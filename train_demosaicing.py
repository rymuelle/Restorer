import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.flop_counter import FlopCounterMode

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from SWLoss.src.slicing_loss import SlicingLoss, GramLoss

from pathlib import Path
from src.training.load_config import load_config
from src.training.losses.CCMLoss import CCMLoss

# Read config
run_config = load_config()
outpath = Path(run_config['numpy_raw_subdir'])
alignment_csv =  outpath / run_config['align_csv']
from src.training.DemoDataset import DemoDataset
from src.Restorer.DemoNAFNet import DemoNAFNet
from src.training.losses.PSNR import PSNRLoss, psnr


CONFIG = {
    "model_name": "Demosaic_resize_2_32_long",
    "experiment_name": "BaseDemosaic",
    "batch_size": 16,
    "lr": 5e-4,
    "sched_end_factor": 1e-6,
    "epochs": 4500,
    "seed": 42,
    "num_workers": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "width": 32,
    "middle_blk_num": 4,
    "enc_blk_nums":[(0, 0), (0, 0)],
    "dec_blk_nums":[(0, 0), (0, 0)],
    "in_channels": 6,
    "lumi_noise": 0,
    "crop_size": 32,
    "residual_mask": False,
    'SWL_scale': 0,
    "iso_range": [0, 1e9],
    "added_noise": 0.,
    "no_raf": False,
    "iter_per_iter": 1,
    "resize_gt": 2,

}

def measure_flops(model, mlflow):
    flop_counter = FlopCounterMode(display=True)
    with flop_counter:
        model(torch.rand(1, 6, 256, 256).to(CONFIG['device']))
    total_flops = flop_counter.get_total_flops()

    mlflow.log_metric("total_flops", total_flops)
        
def train():
    generator = torch.Generator().manual_seed(CONFIG["seed"])


    dataset = DemoDataset("bad_image_csv.csv", validation=False, 
                          crop_size=CONFIG['crop_size'],
                          resize=CONFIG['resize_gt'])
    if CONFIG['no_raf']:
        dataset.csv['raf'] =  dataset.csv.gt_out.str.contains('.raf')
        dataset.csv = dataset.csv[~dataset.csv.raf]
    dataset.csv.reset_index(inplace=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, 
                              generator=generator, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False, 
                            generator=generator, num_workers=CONFIG["num_workers"])

    model = DemoNAFNet(in_channels=CONFIG['in_channels'], width=CONFIG["width"],
                        middle_blk_num=CONFIG["middle_blk_num"], 
                   enc_blk_nums=CONFIG["enc_blk_nums"], dec_blk_nums=CONFIG["dec_blk_nums"],
                     mask=CONFIG['residual_mask'],
                    ).to(CONFIG["device"], )
                   
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=CONFIG["sched_end_factor"], total_iters=CONFIG["epochs"])
    criterion = nn.L1Loss()
    # criterion = CCMLoss()
    texture_criteria = GramLoss(1).to(CONFIG['device'])

    # MLflow Tracking
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=CONFIG["model_name"]):
        # Log Hyperparameters
        mlflow.log_params(CONFIG)
        measure_flops(model, mlflow)
        for epoch in range(CONFIG["epochs"]):
            # Trainig
            model.train()
            train_loss = 0.0
            tloader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
            
            for output in tloader:
                
                
                images, sparse = output['gt'].to(CONFIG["device"]), output['six_chan'].to(CONFIG["device"])
                ccm = output['ccm'].to(CONFIG["device"])
                for _ in range(CONFIG['iter_per_iter']):
                    optimizer.zero_grad()
                    with torch.autocast(device_type=CONFIG["device"], dtype=torch.bfloat16):
                        output = model(sparse)
                        if CONFIG['SWL_scale'] > 0:
                            tloss = texture_criteria(output, images) * CONFIG['SWL_scale']
                        else:
                            tloss = 0 
                        loss = criterion(output, images)
                    loss += tloss
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                    tloader.set_postfix({"loss": f"{loss.item():.4e}"})
            
            avg_train_loss = train_loss / (len(train_set) * CONFIG['iter_per_iter'])
            mlflow.log_metric("train_l1_loss", avg_train_loss, step=epoch)
            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            val_psnr_loss = 0.0
            vloader = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            with torch.no_grad():
                for output in vloader:
                    images, sparse = output['gt'].to(CONFIG["device"]), output['six_chan'].to(CONFIG["device"])
                    ccm = output['ccm'].to(CONFIG["device"])
                    output = model(sparse)
                    loss = criterion(output, images)
                    if CONFIG['SWL_scale'] > 0:
                        tloss = texture_criteria(output, images) * CONFIG['SWL_scale']
                    else:
                        tloss = 0 
                   
                    loss += tloss
                    val_loss += loss.item() * images.size(0)
                    psnr_loss = psnr(output, images) * images.size(0)
                    val_psnr_loss += psnr_loss
            
            avg_val_loss = val_loss / len(val_set)
            avg_val_psnr = val_psnr_loss / len(val_set)
            mlflow.log_metric("val_l1_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_psnr_loss", avg_val_psnr, step=epoch)
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4e} PSNR: {avg_val_psnr:.1f}")

        # Save Artifacts
        mlflow.pytorch.log_model(model, "model")
        
        # Local save as backup
        torch.save(model.state_dict(), f"{CONFIG['model_name']}_final.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()