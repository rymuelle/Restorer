import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from src.training.losses.CombinedPerceptualLoss import VGGPerceptualLoss
import torchvision

class ShadowAwareLoss(nn.Module):
    def __init__(self, 
                 alpha=0.2,
                 beta=5.0,
                 l1_weight=0.16,
                 ssim_weight=0.84,
                 tv_weight=0.0,
                 vgg_loss_weight=0.0,
                 apply_gamma_fn=None,
                 vgg_feature_extractor=None,
                 percept_loss_weight = 0,
                 device=None,
                 sharpness_loss_weight=0):
        """
        Shadow-aware image restoration loss.

        Args:
            alpha: Minimum tone weight for bright pixels.
            beta: Controls how quickly weight increases in shadows.
            l1_weight, ssim_weight, tv_weight, vgg_loss_weight: Loss scaling factors.
            apply_gamma_fn: Optional function to apply gamma correction to input tensors.
            vgg_feature_extractor: Optional VGG feature extractor returning feature maps.
            device: Optional device to move inputs and buffers to.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight
        self.vgg_loss_weight = vgg_loss_weight
        self.apply_gamma_fn = apply_gamma_fn
        self.vfe = vgg_feature_extractor
        self.device = device
        self.percept_loss_weight = percept_loss_weight
        self.VGGPerceptualLoss = VGGPerceptualLoss()
        self.sharpness_loss_weight = sharpness_loss_weight

        if device is not None:
            self.to(device)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] restored image in [0,1]
            target: [B, C, H, W] ground truth in [0,1]
        """
        if self.apply_gamma_fn is not None:
            pred = torch.clamp(pred, 1e-6, 1.0)
            target = torch.clamp(target, 1e-6, 1.0)
            pred = self.apply_gamma_fn(pred).clamp(1e-6, 1)
            target = self.apply_gamma_fn(target).clamp(1e-6, 1)

        # Convert to luminance (BT.709)
        lum = 0.2126 * target[:, 0] + 0.7152 * target[:, 1] + 0.0722 * target[:, 2]  # [B, H, W]
        tone_weight = self.alpha + (1.0 - self.alpha) * torch.exp(-self.beta * lum)
        tone_weight = tone_weight.unsqueeze(1)  # [B, 1, H, W]

        # Weighted L1 loss
        l1 = (tone_weight * torch.abs(pred - target)).mean()

        # Weighted MS-SSIM loss
        ssim = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)

        # TV loss only in low-light areas
        shadow_mask = (lum < 0.2).float().unsqueeze(1)
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        tv = ((shadow_mask[:, :, :, 1:] * dx).mean() +
              (shadow_mask[:, :, 1:, :] * dy).mean())

        # Optional VGG perceptual loss
        vgg_loss_val = 0
        if self.vgg_loss_weight != 0 and self.vfe is not None:
            with torch.no_grad():
                pred_features = self.vfe(pred)
                target_features = self.vfe(target)
            vgg_loss_val = nn.functional.mse_loss(pred_features[0], target_features[0])

        percept_loss = 0
        if self.percept_loss_weight:
            percept_loss = self.VGGPerceptualLoss(pred, target)

        sharpness_loss_value = 0
        if self.sharpness_loss_weight:
            sharpness_loss_value += sharpness_loss(pred, target)

        # Combine weighted terms
        total_loss = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.tv_weight * tv +
            self.vgg_loss_weight * vgg_loss_val +
            self.percept_loss_weight * percept_loss +
            self.sharpness_loss_weight * sharpness_loss_value
        )

        return total_loss

def sharpness_loss(pred, target, loss_func = torch.nn.functional.l1_loss):
    loss = loss_func(pred, target)
    pred_prime = torchvision.transforms.functional.gaussian_blur(pred, kernel_size=[5, 5], sigma=[1.0, 1.0])
    target_prime = torchvision.transforms.functional.gaussian_blur(target, kernel_size=[5, 5], sigma=[1.0, 1.0])
    loss += loss_func(pred-pred_prime, target-target_prime)
    return loss
