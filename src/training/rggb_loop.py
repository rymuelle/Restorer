from time import perf_counter
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from src.training.utils import apply_gamma_torch

def make_conditioning(conditioning, device):
    B = conditioning.shape[0]
    conditioning_extended = torch.zeros(B, 1).to(device)
    conditioning_extended[:, 0] = conditioning[:, 0]
    return conditioning_extended


def train_one_epoch_rggb(epoch, _model, _optimizer, _outname, _loader, _device, _loss_func, _clipping, log_interval = 10, sleep=0.2):
    _model.train()
    total_loss, n_images, total_final_image_loss = 0.0, 0, 0.0
    start = perf_counter()
    pbar = tqdm(enumerate(_loader), total=len(_loader), desc=f"Train Epoch {epoch}")

    for batch_idx, (output) in pbar:
    # for output in train_loader:
        noisy = output['noisy'].float().to(_device)
        conditioning = output['conditioning'].float().to(_device)
        gt = output['aligned'].float().to(_device)
        rggb = output['rggb'].float().to(_device)

        conditioning = make_conditioning(conditioning, _device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            pred = _model(rggb, conditioning, noisy) 

        loss = _loss_func(pred, gt)
        _optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_model.parameters(), _clipping)
        _optimizer.step()

        total_loss +=  float(loss.detach().cpu())
        n_images += gt.shape[0]

        # Testing final image quality
        final_image_loss = nn.functional.l1_loss(pred, gt)
        total_final_image_loss += final_image_loss.item()
        del loss, pred, final_image_loss
        torch.mps.empty_cache() 

        if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({"loss": f"{total_loss/n_images:.4f}"})

        time.sleep(sleep)

    torch.save(_model.state_dict(), _outname)

    print(f"[Epoch {epoch}] "
                f"Train loss: {total_loss/n_images:.6f} "
                f"Final image val loss: {total_final_image_loss/n_images:.6f} "
                f"Time: {perf_counter()-start:.1f}s "
                f"Images: {n_images}")

    return total_loss / max(1, n_images), perf_counter()-start




def visualize(idxs, _model, dataset, _device, _loss_func):
    import matplotlib.pyplot as plt
    _model.train()
    total_loss, n_images, total_final_image_loss = 0.0, 0, 0.0
    start = perf_counter()


    for idx in idxs:
    # for output in train_loader:
        row = dataset[idx]
        noisy = row['noisy'].unsqueeze(0).float().to(_device)
        conditioning = row['conditioning'].float().unsqueeze(0).to(_device)
        gt = row['aligned'].unsqueeze(0).float().to(_device)
        rggb = row['rggb'].unsqueeze(0).float().to(_device)

        conditioning = make_conditioning(conditioning, _device)
        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                pred = _model(rggb, conditioning, noisy) 
        loss = _loss_func(pred, gt)


        total_loss +=  float(loss.detach().cpu())
        n_images += gt.shape[0]

        # Testing final image quality
        final_image_loss = nn.functional.l1_loss(pred, gt)
        total_final_image_loss += final_image_loss.item()

        plt.subplots(2, 2, figsize=(15, 15))

        plt.subplot(2, 2, 1)
        pred = apply_gamma_torch(pred[0].cpu().permute(1, 2, 0))
        plt.imshow(pred)

        plt.subplot(2, 2, 2)
        noisy = apply_gamma_torch(noisy[0].cpu().permute(1, 2, 0))
        plt.imshow(noisy)

        plt.subplot(2, 2, 3)
        gt = apply_gamma_torch(gt[0].cpu().permute(1, 2, 0))
        plt.imshow(gt)

        plt.subplot(2, 2, 4)
        plt.imshow(pred - gt + 0.5)
        plt.show()
        plt.clf()

    n_images = len(idxs)
    print(
                f"Train loss: {total_loss/n_images:.6f} "
                f"Final image val loss: {total_final_image_loss/n_images:.6f} "
                f"Time: {perf_counter()-start:.1f}s "
                f"Images: {n_images}")

    return total_loss / max(1, n_images), perf_counter()-start