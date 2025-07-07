import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.layers import DropPath


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta
        return x


class FiLMGenerator(nn.Module):
    def __init__(self, iso_dim, film_dims):
        super().__init__()
        self.output_layer = nn.Linear(64, sum(2 * d for d in film_dims))
        self.mlp = nn.Sequential(nn.Linear(iso_dim, 64), nn.ReLU(), self.output_layer)
        self.film_dims = film_dims

        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.log_power = nn.Parameter(torch.tensor(0.0))  # log(1.0)

        # Custom init
        with torch.no_grad():
            start = 0
            for d in film_dims:
                # gamma
                self.output_layer.bias[start : start + d].fill_(1.0)
                self.output_layer.weight[start : start + d].zero_()
                start += d
                # beta
                self.output_layer.bias[start : start + d].zero_()
                self.output_layer.weight[start : start + d].zero_()
                start += d

    def forward(self, iso):
        iso = self.gamma * iso ** self.log_power.exp() + self.beta
        out = self.mlp(iso)  # shape [B, total_params]
        gammas_betas = torch.split(out, [2 * d for d in self.film_dims], dim=1)
        results = [
            (gb[:, :d], gb[:, d:]) for gb, d in zip(gammas_betas, self.film_dims)
        ]
        return results  # List of (gamma_i, beta_i)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionedChannelAttention(nn.Module):
    def __init__(self, dims, cat_dims):
        super().__init__()
        in_dim = dims + cat_dims
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_dim, int(in_dim*1.5)),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(int(in_dim*1.5), dims)
        # )
        self.mlp = nn.Sequential(nn.Linear(in_dim, dims))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca


# Layernorm from CGnet
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# handle multiple input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) is tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f"Done image [{i + 1:<3}/ {max_iter}], "
                    f"fps: {fps:.1f} img / s, "
                    f"times per image: {1000 / fps:.1f} ms / img",
                    flush=True,
                )

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f"Overall fps: {fps:.1f} img / s, "
                f"times per image: {1000 / fps:.1f} ms / img",
                flush=True,
            )
            break
    return fps


class ConditionedChannelAttentionCN(nn.Module):
    def __init__(self, dims, cat_dims):
        super().__init__()
        in_dim = dims + cat_dims
        self.mlp = nn.Sequential(nn.Linear(in_dim, dims))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        x = x.permute(0, 3, 1, 2)
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels)
        return ca


class SimpleGateCN(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=3)
        return F.gelu(x1) * F.sigmoid(x2)


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, cond_chans=0, expand_dim=4):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expand_dim * dim)  # pointwise conv

        self.sg = SimpleGateCN()
        self.sca = ConditionedChannelAttentionCN(expand_dim * dim, cond_chans)
        # self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(expand_dim * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, inp):
        x, cond = inp
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        # x = self.sg(x)
        x = F.gelu(x)
        x = x * self.sca(x, cond)
        # x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x, cond
