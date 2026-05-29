import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Domain/Project-specific Imports
from SWLoss.src.slicing_loss import SlicingLoss, GramLoss
from src.training.load_config import load_config
from src.training.losses.CCMLoss import CCMLoss
from src.training.JDDDataset import JDDDataset
from src.training.losses.PSNR import PSNRLoss, psnr

# Restorer Model Registry
from src.Restorer.DemoRestormer import DemoRestormer
from src.Restorer.DemoNAFNetMamba import DemoNAFNetMamba
from src.Restorer.DemoNAFNetEAMamba import DemoNAFNetEAMamba
from src.Restorer.DemoNAFNetDIT import DemoNAFNetDIT
from src.Restorer.DemoNAFNet import DemoNAFNet

CONFIG = {
    "model_name": "DemoNAFNetDIT",
    "experiment_name": "BaseDenoising",
    "lr": 5e-4,
    "sched_end_factor": 1e-6,
    "epochs": 1000,
    "seed": 42,
    "num_workers": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "width": 32,
    "middle_blk_num": (2, 0),
    "enc_blk_nums":[(1, 0), (1, 0), (2, 0), (2, 0)],
    "dec_blk_nums":[(2, 0), (2, 0), (1, 0), (1, 0)],
    
    "num_heads": 4,
    "in_channels": 6,
    "lumi_noise": 0,
    "residual_mask": False,
    'SWL_scale': 0,
    "iso_range": [0, 1e9],
    "added_noise": 0.,
    "no_raf": True,
    "iter_per_iter": 1,
    "CSV": "refit.csv",
    "gb_filter": .1,
    "model": "DemoNAFNetDIT"
}

TRAINING_SCHEDULE = [
    {"start_epoch": 0,   "crop_size": 80,  "batch_size": 16},
    {"start_epoch": int(.5*CONFIG['epochs']), "crop_size": 128, "batch_size": 16},
    {"start_epoch": int(.8*CONFIG['epochs']), "crop_size": 256, "batch_size": 16},

]
print(TRAINING_SCHEDULE)


def measure_flops(model, device, crop_size, epoch):
    """Measures total FLOPs based on the current scheduled crop size."""
    flop_counter = FlopCounterMode(display=False)
    dummy_input = torch.rand(1, CONFIG['in_channels'], crop_size, crop_size, device=device)
    
    with flop_counter:
        model(dummy_input)
        
    total_flops = flop_counter.get_total_flops()
    mlflow.log_metric("total_flops", total_flops, step=epoch)
    print(f"Profiled FLOPs for crop size {crop_size}x{crop_size}: {total_flops:,}")


def model_factory(model_name, config, device):
    """Encapsulates model initialization logic cleanly."""
    models_map = {
        "DemoRestormer": DemoRestormer,
        "DemoNAFNetMamba": DemoNAFNetMamba,
        "DemoNAFNetEAMamba": DemoNAFNetEAMamba,
        "DemoNAFNetDIT": DemoNAFNetDIT,
        "DemoNAFNet": DemoNAFNet
    }
    
    model_class = models_map.get(model_name, DemoNAFNet)
    kwargs = {
        "in_channels": config['in_channels'],
        "width": config["width"],
        "middle_blk_num": config["middle_blk_num"],
        "enc_blk_nums": config["enc_blk_nums"],
        "dec_blk_nums": config["dec_blk_nums"],
        "mask": config['residual_mask']
    }
    
    if model_name in ["DemoRestormer", "DemoNAFNetDIT"]:
        kwargs["num_heads"] = config['num_heads']
        
    return model_class(**kwargs).to(device)


def prepare_datasets(config, generator):
    """Instantiates base datasets with a placeholder crop size (updated dynamically later)."""
    train_dataset = JDDDataset(config['CSV'], validation=False, crop_size=64)
    val_dataset = JDDDataset(config['CSV'], validation=True, crop_size=64)

    for dataset in [train_dataset, val_dataset]:
        dataset.csv = dataset.csv[~dataset.csv.bad]
        dataset.csv = dataset.csv[
            (dataset.csv.iso >= config['iso_range'][0]) & 
            (dataset.csv.iso <= config['iso_range'][1])
        ]
        dataset.csv = dataset.csv[(dataset.csv.gb - 1).abs() < config['gb_filter']]

        if config['no_raf']:
            dataset.csv['raf'] = dataset.csv.gt_out.str.contains('.raf')
            dataset.csv = dataset.csv[~dataset.csv.raf]
        dataset.csv.reset_index(inplace=True)

    dataset_length = len(train_dataset)
    train_size = int(0.8 * dataset_length)
    
    indices = torch.randperm(dataset_length, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    
    return train_set, val_set


def train():
    device = torch.device(CONFIG["device"])
    generator = torch.Generator().manual_seed(CONFIG["seed"])

    # Setup Dataset Subsets
    train_set, val_set = prepare_datasets(CONFIG, generator)
    
    # Initialize Core Engine Components
    model = model_factory(CONFIG['model'], CONFIG, device)
                   
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=CONFIG["sched_end_factor"], total_iters=CONFIG["epochs"]
    )
    criterion = nn.L1Loss()
    texture_criteria = SlicingLoss(1).to(device)

    mlflow.set_experiment(CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=CONFIG["model_name"]):
        mlflow.log_params(CONFIG)
        
        current_stage = None
        train_loader, val_loader = None, None

        for epoch in range(CONFIG["epochs"]):
            
            active_stage = None
            for stage in TRAINING_SCHEDULE:
                if epoch >= stage["start_epoch"]:
                    active_stage = stage
            
            # Re-initialize data pipelines if we entered a new stage boundary
            if active_stage != current_stage:
                current_stage = active_stage
                print(f"\nSWAPPING STAGE at Epoch {epoch} -> Crop Size: {current_stage['crop_size']}, Batch Size: {current_stage['batch_size']}")
                
                # Update underlying dataset state configurations
                train_set.dataset.crop_size = current_stage["crop_size"]
                val_set.dataset.crop_size = current_stage["crop_size"]
                
                # Rebuild data loaders with the new batch size specifications
                train_loader = DataLoader(
                    train_set, batch_size=current_stage["batch_size"], shuffle=True, 
                    generator=generator, num_workers=CONFIG["num_workers"], pin_memory=True
                )
                val_loader = DataLoader(
                    val_set, batch_size=current_stage["batch_size"], shuffle=False, 
                    generator=generator, num_workers=CONFIG["num_workers"], pin_memory=True
                )
                
                # Re-profile FLOPs since spatial scales changed
                measure_flops(model, device, current_stage["crop_size"], epoch)
                mlflow.log_metric("scheduled_crop_size", current_stage["crop_size"], step=epoch)
                mlflow.log_metric("scheduled_batch_size", current_stage["batch_size"], step=epoch)

            # ---------------------------------------------------------
            # Training Phase
            # ---------------------------------------------------------
            model.train()
            train_loss = 0.0
            tloader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
            
            for batch in tloader:
                images = batch['aligned'].to(device, non_blocking=True)
                sparse = batch['deg'].to(device, non_blocking=True)
                mono_noise = batch['mono_noise_proportion'].to(CONFIG["device"])

                ccm = batch['ccm'].to(CONFIG["device"])
                images = (images * (1 + mono_noise * CONFIG['added_noise'])).clamp(0,1)
                for _ in range(CONFIG['iter_per_iter']):
                    optimizer.zero_grad()
                    
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        output = model(sparse)
                        tloss = texture_criteria(output, images) * CONFIG['SWL_scale'] if CONFIG['SWL_scale'] > 0 else 0
                        loss = criterion(output, images) + tloss
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                    tloader.set_postfix({"loss": f"{loss.item():.4e}"})
            
            avg_train_loss = train_loss / (len(train_set) * CONFIG['iter_per_iter'])
            mlflow.log_metric("train_l1_loss", avg_train_loss, step=epoch)
            scheduler.step()

            # ---------------------------------------------------------
            # Validation Phase
            # ---------------------------------------------------------
            model.eval()
            val_loss = 0.0
            val_psnr_loss = 0.0
            vloader = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            with torch.no_grad():
                for batch in vloader:
                    images = batch['aligned'].to(device, non_blocking=True)
                    sparse = batch['deg'].to(device, non_blocking=True)
                    mono_noise = batch['mono_noise_proportion'].to(CONFIG["device"])

                    ccm = batch['ccm'].to(CONFIG["device"])
                    images = (images * (1 + mono_noise * CONFIG['added_noise'])).clamp(0,1)
                    
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        output = model(sparse)
                        tloss = texture_criteria(output, images) * CONFIG['SWL_scale'] if CONFIG['SWL_scale'] > 0 else 0
                        loss = criterion(output, images) + tloss
                   
                    val_loss += loss.item() * images.size(0)
                    val_psnr_loss += psnr(output, images) * images.size(0)
            
            avg_val_loss = val_loss / len(val_set)
            avg_val_psnr = val_psnr_loss / len(val_set)
            
            mlflow.log_metric("val_l1_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_psnr_loss", avg_val_psnr, step=epoch)
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4e} PSNR: {avg_val_psnr:.1f}")

        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), f"{CONFIG['model_name']}_final.pth")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()