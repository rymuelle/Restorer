import torchvision.models as models
import torchvision
import torch.nn as nn
import torch

from pytorch_msssim import ms_ssim
from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d

import torch.nn.functional as F

import torch_dct as dct_2d
import lpips



class CombinedPerceptualLoss(nn.Module):
    def __init__(
        self,
        lambda_l1=0,
        lambda_invariant_l1=0,
        lambda_ssim=0.0,
        lambda_vgg=0.0,

        lambda_frequency=0.0,
        lambda_chroma=0.0,
        lambda_luma=0.0,
        lambda_luma_mse=0.0,
        lambda_luma_struct=0.0,
        lambda_lpips=0.0,
        lambda_sharpness_loss=0.0,
        lambda_conservative_l1=0.0,
        lambda_ssim_coarse=0.0,
        lambda_ssim_fine=0.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_invariant_l1 = lambda_invariant_l1

        self.lambda_ssim = lambda_ssim
        self.lambda_ssim_coarse = lambda_ssim_coarse
        self.lambda_ssim_fine = lambda_ssim_fine
        self.lambda_vgg = lambda_vgg
        self.lambda_frequency = lambda_frequency
        self.lambda_chroma = lambda_chroma
        self.lambda_luma = lambda_luma
        self.lambda_luma_mse = lambda_luma_mse
        self.lambda_luma_struct = lambda_luma_struct
        self.lambda_lpips = lambda_lpips
        self.lambda_sharpness_loss = lambda_sharpness_loss
        self.lambda_conservative_l1 = lambda_conservative_l1

        self.vgg_loss = VGGPerceptualLoss()
        self.l1_loss = nn.L1Loss()
        self.frequency_loss = FrequencyLoss()
        self.chroma_loss = ChromaLoss(loss_func=F.mse_loss)
        self.luma_loss = LumaLoss()
        self.luma_loss_mse = LumaLoss(loss_func=F.mse_loss)
        self.luma_loss_struct = LumaLoss(loss_func=structural_loss)
        self.lpips_loss = lpips.LPIPS(net="vgg")
        self.sharpness_loss = sharpness_loss
        self.conservative_l1_loss = conservative_l1

    def forward(self, output, inp, target):
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
                1
                - ms_ssim(
                    output,
                    target,
                    data_range=1.0,
                    size_average=True,
                    weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                )
            )
        if self.lambda_ssim_coarse > 0:
            # [0, 0.298964, 0.314037, 0.247357, 0.139537]
            weights = [0.1, 0.2, 0.448028, 0.352898, 0.199074]
            loss += self.lambda_ssim_coarse * (
                1
                - ms_ssim(
                    output, target, data_range=1.0, size_average=True, weights=weights
                )
            )
        if self.lambda_ssim_fine > 0:
            loss += self.lambda_ssim_fine * (
                1
                - ms_ssim(
                    output,
                    target,
                    data_range=1.0,
                    size_average=True,
                    weights=[1, 0, 0, 0, 0],
                )
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
        if self.lambda_luma > 0:
            loss += self.lambda_luma * self.luma_loss(output, target)
        if self.lambda_luma_mse > 0:
            loss += self.lambda_luma_mse * self.luma_loss_mse(output, target)
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
        if self.lambda_conservative_l1 > 0:
            loss += self.lambda_conservative_l1 * self.conservative_l1_loss(
                output, inp, target
            )
        if self.lambda_luma_struct > 0:
            loss += self.lambda_luma_struct * self.luma_loss_struct(output, target)
        return loss


class VGGPerceptualLoss(nn.Module):
    """
    Runs prediction and target through VGG and compares the output of each layer to compute a perceptual loss.
    """

    def __init__(self, layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        vgg = models.vgg16(weights=True).features[: max(layers) + 1]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.layers = layers
        self.weights = weights
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        wdix = 0
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if idx in self.layers:
                loss += self.criterion(x, y) * self.weights[wdix]
                wdix += 1
        return loss 



def gram_matrix(features):
    """Compute the Gram matrix from feature maps."""
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)  # Flatten spatial dimensions
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    gram = gram / (C * H * W)  # Normalize
    return gram

class StyleLoss(nn.Module):
    def __init__(self, target_features):
        """
        target_features: List of feature maps from the style image at chosen layers.
        """
        super().__init__()
        self.target_grams = [gram_matrix(f).detach() for f in target_features]

    def forward(self, input_features):
        """
        input_features: List of feature maps from the generated image at the same layers.
        """
        loss = 0
        for inp_feat, target_gram in zip(input_features, self.target_grams):
            G = gram_matrix(inp_feat)
            loss += nn.functional.mse_loss(G, target_gram)
        return loss

class VGGStyleFeatureExtractor(nn.Module):
    def __init__(self, layers=None):
        """
        Extracts features from specific VGG19 layers for style loss.
        Default layers match those used in Gatys et al.'s style transfer.
        """
        super().__init__()
        if layers is None:
            # Default layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                      '19': 'conv4_1', '28': 'conv5_1'}

        self.vgg = models.vgg19(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.selected_layers = layers

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features
    
    
# def rgb_to_ycbcr(img):
#     r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
#     cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5
#     return torch.cat([y, cb, cr], dim=1)


# def ycbcr_to_rgb(img):
#     y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5

#     r = y + 1.402 * cr
#     g = y - 0.344136 * cb - 0.714136 * cr
#     b = y + 1.772 * cb

#     return torch.cat([r, g, b], dim=1)

# def rgb_to_ycbcr_jpeg(img):
#     r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#     y  = 0.299 * r + 0.587 * g + 0.114 * b
#     cb = (b - y) * 0.564 + 0.5
#     cr = (r - y) * 0.713 + 0.5
#     return torch.cat([y, cb, cr], dim=1)

# def ycbcr_to_rgb_jpeg(img):
#     y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5
#     r = y + 1.403 * cr
#     g = y - 0.344 * cb - 0.714 * cr
#     b = y + 1.770 * cb
#     return torch.cat([r, g, b], dim=1)

from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb

class ChromaLoss(torch.nn.Module):
    def __init__(self, loss_func=F.l1_loss):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, output, target):
        ycbcr_out = rgb_to_ycbcr(output)
        ycbcr_tgt = rgb_to_ycbcr(target)
        return self.loss_func(ycbcr_out[:, 1:], ycbcr_tgt[:, 1:])  # Only CB + CR


class LumaLoss(torch.nn.Module):
    def __init__(self, loss_func=F.l1_loss):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, output, target):
        ycbcr_out = rgb_to_ycbcr(output)
        ycbcr_tgt = rgb_to_ycbcr(target)
        return self.loss_func(ycbcr_out[:, :1], ycbcr_tgt[:, :1])  # Only Y


class FrequencyLoss(torch.nn.Module):
    def forward(self, output, target):
        dct_out = dct_2d(output)
        dct_gt = dct_2d(target)
        return F.l1_loss(dct_out, dct_gt)


def sharpness_loss(pred, target):
    loss_func = F.l1_loss
    loss = loss_func(pred, target)
    pred_prime = torchvision.transforms.functional.gaussian_blur(
        pred, kernel_size=[11, 11], sigma=[1.0, 1.0]
    )
    target_prime = torchvision.transforms.functional.gaussian_blur(
        target, kernel_size=[11, 11], sigma=[1.0, 1.0]
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
