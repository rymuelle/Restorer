import torchvision.models as models
import torchvision
import torch.nn as nn
import torch

from pytorch_msssim import ms_ssim
import torch.nn.functional as F

import torch_dct as dct_2d
import lpips


class CombinedPerceptualLoss(nn.Module):
    def __init__(
        self,
        lambda_l1=1,
        lambda_invariant_l1=0,
        lambda_ssim=0.0,
        lambda_vgg=0.0,
        lambda_frequency=0.0,
        lambda_chroma=0.0,
        lambda_lpips=0.0,
        lambda_sharpness_loss=0.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_invariant_l1 = lambda_invariant_l1

        self.lambda_ssim = lambda_ssim
        self.lambda_vgg = lambda_vgg
        self.lambda_frequency = lambda_frequency
        self.lambda_chroma = lambda_chroma
        self.lambda_lpips = lambda_lpips
        self.lambda_sharpness_loss = lambda_sharpness_loss

        self.vgg_loss = VGGPerceptualLoss()
        self.l1_loss = nn.L1Loss()
        self.frequency_loss = FrequencyLoss()
        self.chroma_loss = ChromaLoss()
        self.lpips_loss = lpips.LPIPS(net="vgg")
        self.sharpness_loss = sharpness_loss

    def forward(self, output, target):
        # Assumes output and target are gamma-compressed RGB [B, 3, H, W]
        loss = 0
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.l1_loss(output, target)
        if self.lambda_invariant_l1 > 0:
            mean_target = target.mean(dim=(1, 2, 3))
            mean_output = output.mean(dim=(1, 2, 3))
            loss += self.lambda_invariant_l1 * self.l1_loss(
                output - mean_output, target - mean_target
            )
        if self.lambda_ssim > 0:
            loss += self.lambda_ssim * (
                1 - ms_ssim(output, target, data_range=1.0, size_average=True)
            )
        if self.lambda_vgg > 0:
            vgg_output = apply_gamma(output)
            vgg_output = transform_vgg(vgg_output)
            vgg_target = apply_gamma(target)
            vgg_target = transform_vgg(vgg_target)
            loss += self.lambda_vgg * self.vgg_loss(vgg_output, vgg_target)
        if self.lambda_frequency > 0:
            loss += self.lambda_frequency * self.frequency_loss(output, target)
        if self.lambda_chroma > 0:
            loss += self.lambda_chroma * self.chroma_loss(output, target)
        if self.lambda_lpips > 0:
            _output = output.clone()
            _target = target.clone()
            _output[_output < 0] = 0
            _output[_output > 1] = 1
            _target[_target < 0] = 0
            _target[_target > 1] = 1
            _target = apply_gamma(_target)
            _output = apply_gamma(_output)
            _output = (output - 0.5) / 0.5
            _target = (target - 0.5) / 0.5
            lpips = self.lpips_loss(_output, _target).mean()
            loss += self.lambda_lpips * lpips
        if self.lambda_sharpness_loss > 0:
            loss += self.lambda_sharpness_loss * self.sharpness_loss(output, target)
        return loss


class VGGPerceptualLoss(nn.Module):
    """
    Runs prediction and target through VGG and compares the output of each layer to compute a perceptual loss.
    """

    def __init__(self, layers=[3, 8, 15, 22], weight=1.0):
        super().__init__()
        vgg = models.vgg16(weights=True).features[: max(layers) + 1]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.layers = layers
        self.weight = weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += self.criterion(x, y)
        return loss * self.weight


def rgb_to_ycbcr(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


class ChromaLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ycbcr_out = rgb_to_ycbcr(output)
        ycbcr_tgt = rgb_to_ycbcr(target)
        return F.l1_loss(ycbcr_out[:, 1:], ycbcr_tgt[:, 1:])  # Only CB + CR


class FrequencyLoss(torch.nn.Module):
    def forward(self, output, target):
        dct_out = dct_2d(output)
        dct_gt = dct_2d(target)
        return F.l1_loss(dct_out, dct_gt)


def sharpness_loss(pred, target):
    loss_func = F.l1_loss
    loss = loss_func(pred, target)
    pred_prime = torchvision.transforms.functional.gaussian_blur(
        pred, kernel_size=[5, 5], sigma=[1.0, 1.0]
    )
    target_prime = torchvision.transforms.functional.gaussian_blur(
        target, kernel_size=[5, 5], sigma=[1.0, 1.0]
    )
    loss += loss_func(pred - pred_prime, target - target_prime) * 2
    return loss


def transform_vgg(input):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return (input - IMAGENET_MEAN) / IMAGENET_STD


def apply_gamma(_tensor):
    tensor = _tensor.clone()
    img_mask = tensor > 0.0031308
    tensor[img_mask] = 1.055 * torch.pow(tensor[img_mask], 1.0 / 2.4) - 0.055
    tensor[~img_mask] *= 12.92
    return tensor
