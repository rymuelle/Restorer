from time import perf_counter
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from src.training.utils import apply_gamma_torch
import mlflow

def make_conditioning(conditioning, device):
    B = conditioning.shape[0]
    conditioning_extended = torch.zeros(B, 1).to(device)
    conditioning_extended[:, 0] = conditioning[:, 0]
    return conditioning_extended


def train_one_epoch(epoch, _model, _optimizer, _loader, _device, _loss_func, _clipping, 
                    log_interval = 10, sleep=0.0, rggb=False,
                    max_batches=0):
    _model.train()
    total_loss, n_images, total_l1_loss = 0.0, 0, 0.0
    start = perf_counter()
    pbar = tqdm(enumerate(_loader), total=len(_loader), desc=f"Train Epoch {epoch}")

    for batch_idx, (output) in pbar:
        noisy = output['noisy'].float().to(_device)
        conditioning = output['conditioning'].float().to(_device)
        gt = output['aligned'].float().to(_device)
        input = output['sparse'].float().to(_device)
        if rggb:
            input = output['rggb'].float().to(_device)
        conditioning = make_conditioning(conditioning, _device)

        _optimizer.zero_grad(set_to_none=True)
        pred = _model(input, conditioning, noisy) 

        loss = _loss_func(pred, gt)
        _optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_model.parameters(), _clipping)
        _optimizer.step()

        total_loss +=  float(loss.detach().cpu())
        n_images += gt.shape[0]

        # Testing final image quality
        final_image_loss = float(nn.functional.l1_loss(pred, gt).detach().cpu())
        total_l1_loss += final_image_loss
        del loss, pred, final_image_loss
        torch.mps.empty_cache() 

        if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({"loss": f"{total_loss/n_images:.4f}"})

        if (max_batches > 0) and (batch_idx+1 > max_batches): break
        time.sleep(sleep)

    train_time = perf_counter()-start
    print(f"[Epoch {epoch}] "
                f"Train loss: {total_loss/n_images:.6f} "
                f"L1 loss: {total_l1_loss/n_images:.6f} "
                f"Time: {train_time:.1f}s "
                f"Images: {n_images}")
    mlflow.log_metric("train_loss", total_loss/n_images, step=epoch)
    mlflow.log_metric("l1_loss", total_l1_loss/n_images, step=epoch)
    mlflow.log_metric("epoch_duration_s", train_time, step=epoch)
    mlflow.log_metric("learning_rate", _optimizer.param_groups[0]['lr'], step=epoch)

    return total_loss / max(1, n_images), perf_counter()-start




def visualize(idxs, _model, dataset, _device, _loss_func, rggb=False):
    import matplotlib.pyplot as plt
    _model.train()
    total_loss, n_images, total_final_image_loss = 0.0, 0, 0.0
    start = perf_counter()

    for idx in idxs:
        row = dataset[idx]
        noisy = row['noisy'].unsqueeze(0).float().to(_device)
        conditioning = row['conditioning'].float().unsqueeze(0).to(_device)
        gt = row['aligned'].unsqueeze(0).float().to(_device)
        input = row['sparse'].unsqueeze(0).float().to(_device)
        if rggb:
            input = row['rggb'].unsqueeze(0).float().to(_device)

        conditioning = make_conditioning(conditioning, _device)
        
        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                pred = _model(input, conditioning, noisy) 
        loss = _loss_func(pred, gt)

        total_loss +=  float(loss.detach().cpu())
        n_images += gt.shape[0]
        # Testing final image quality
        final_image_loss = nn.functional.l1_loss(pred, gt)
        total_final_image_loss += final_image_loss.item()

        plt.subplots(2, 3, figsize=(30, 15))

        plt.subplot(2, 3, 1)
        pred = apply_gamma_torch(pred[0].cpu().permute(1, 2, 0))
        plt.imshow(pred)

        plt.subplot(2, 3, 2)
        noisy = apply_gamma_torch(noisy[0].cpu().permute(1, 2, 0))
        plt.imshow(noisy)

        plt.subplot(2, 3, 3)
        gt = apply_gamma_torch(gt[0].cpu().permute(1, 2, 0))
        plt.imshow(gt)

        plt.subplot(2, 3, 4)
        plt.imshow(pred - gt + 0.5)


        plt.subplot(2, 3, 5)
        plt.imshow(noisy - pred + 0.5)

        plt.subplot(2, 3, 6)
        plt.imshow(noisy - gt + 0.5)
        plt.show()
        plt.clf()

    n_images = len(idxs)
    print(
                f"Train loss: {total_loss/n_images:.6f} "
                f"Final image val loss: {total_final_image_loss/n_images:.6f} "
                f"Time: {perf_counter()-start:.1f}s "
                f"Images: {n_images}")

    return total_loss / max(1, n_images), total_final_image_loss / max(1, n_images), perf_counter()-start