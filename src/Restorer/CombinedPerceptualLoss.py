import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from pytorch_msssim import ms_ssim
from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
import lpips
import torch_dct as dct_2d
from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb


# Losses

class FrequencyLoss(nn.Module):
    def forward(self, output, target):
        return F.l1_loss(dct_2d.dct_2d(output), dct_2d.dct_2d(target))


class ChromaLoss(nn.Module):
    def __init__(self, loss_func=F.l1_loss):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, output, target):
        return self.loss_func(rgb_to_ycbcr(output)[:, 1:], rgb_to_ycbcr(target)[:, 1:])


class LumaLoss(nn.Module):
    def __init__(self, loss_func=F.l1_loss):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, output, target):
        return self.loss_func(rgb_to_ycbcr(output)[:, :1], rgb_to_ycbcr(target)[:, :1])


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28], weights=[1.0]*5, apply_gamma_curve=False):
        super().__init__()
        vgg = models.vgg16(weights=True).features[: max(layers) + 1]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.layers = layers
        self.weights = weights
        self.criterion = nn.L1Loss()
        self.apply_gamma_curve = apply_gamma_curve

    def forward(self, pred, target):
        if self.apply_gamma_curve:
            pred = apply_gamma(pred)
            target = apply_gamma(target)
        loss = 0.0
        x = pred
        y = target
        wdix = 0
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if idx in self.layers:
                loss +=  self.weights[wdix]* self.criterion(x, y)
                wdix += 1
        return loss 
    

def gram_matrix(features):
    """Compute the Gram matrix from feature maps."""
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)  # Flatten spatial dimensions
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    gram = gram / (C * H * W)  # Normalize
    return gram

class StyleLoss(VGGPerceptualLoss):
    def forward(self, pred, target):
        if self.apply_gamma_curve:
            pred = apply_gamma(pred)
            target = apply_gamma(target)
        loss = 0.0
        x = pred
        y = target
        wdix = 0
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if idx in self.layers:
                x_gram = gram_matrix(x)
                y_gram = gram_matrix(y)
                loss += self.weights[wdix] * self.criterion(x_gram, y_gram)
                wdix += 1
        return loss




def transform_vgg(input):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (input - mean) / std


def apply_gamma(_tensor):
    tensor = _tensor.clone()
    img_mask = tensor > 0.0031308
    tensor[img_mask] = 1.055 * torch.pow(tensor[img_mask], 1.0 / 2.4) - 0.055
    tensor[~img_mask] *= 12.92
    return tensor

def sharpness_loss(pred, target):
    pred_blur = torchvision.transforms.functional.gaussian_blur(pred, 11, sigma=1.0)
    target_blur = torchvision.transforms.functional.gaussian_blur(target, 11, sigma=1.0)
    return  F.l1_loss(pred - pred_blur, target - target_blur)


def conservative_l1(pred, inp, gt, sout=1, sin=1 / 10):
    same_side = (pred - gt) * (inp - gt) > 0
    between = (pred - gt) * (pred - inp) < 0
    diff = (pred - gt).abs()
    diff[between] *= sin
    diff[~same_side] *= sout
    diff[same_side] *= sout
    diff[same_side * ~between] += (gt - inp).abs()[same_side * ~between] * (sin - 1)
    return diff.mean()



def structural_loss(pred, gt):
    win_size, win_sigma = 11, 1.5
    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([pred.shape[1]] + [1] * (len(pred.shape) - 1))
    _, cs = _ssim(pred, gt, data_range=1.0, win=win)
    return 1 - cs.mean()

def gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return torch.mean(torch.abs(pred_dx - gt_dx)) + torch.mean(
        torch.abs(pred_dy - gt_dy)
    )

# ==== Combined Loss Class ====

class CombinedPerceptualLoss(nn.Module):
    def __init__(self, **lambdas):
        super().__init__()
        self.loss_weights = lambdas

        # Register loss components
        self.loss_modules = {
            "l1": nn.L1Loss(),
            "mse": nn.MSELoss(),
            "ssim": lambda p, t: 1 - ms_ssim(p, t, data_range=1.0, size_average=True),
            "ssim_coarse": lambda p, t: 1 - ms_ssim(p, t, data_range=1.0, size_average=True,
                                                    weights=[0.1, 0.2, 0.448028, 0.352898, 0.199074]),
            "ssim_fine": lambda p, t: 1 - ms_ssim(p, t, data_range=1.0, size_average=True,
                                                  weights=[1, 0, 0, 0, 0]),
            "vgg": VGGPerceptualLoss(apply_gamma_curve=True),
            "style": StyleLoss(layers=[0, 5, 10, 17], weights=[1.0, 1.0, 0.5, 0.25], apply_gamma_curve=True),
            "frequency": FrequencyLoss(),
            "chroma": ChromaLoss(),
            "luma": LumaLoss(),
            "luma_mse": LumaLoss(loss_func=F.mse_loss),
            "luma_struct": LumaLoss(loss_func=structural_loss),
            "lpips": lpips.LPIPS(net="vgg"),
            "sharpness_loss": sharpness_loss,
        }

    def forward(self, pred, target):
        loss = 0.0
        for name, weight in self.loss_weights.items():
            if weight <= 0:
                continue
            if name == "invariant_l1":
                mean_t = target.mean(dim=(1, 2, 3), keepdim=True)
                mean_p = pred.mean(dim=(1, 2, 3), keepdim=True)
                val = self.loss_modules[name](pred - mean_p, target - mean_t)
            elif name == "lpips":
                p = apply_gamma(pred).clamp(0, 1) * 2 - 1
                t = apply_gamma(target).clamp(0, 1) * 2 - 1
                val = self.loss_modules[name](p, t).mean()
            else:
                val = self.loss_modules[name](pred, target)
            loss += weight * val

        return loss
