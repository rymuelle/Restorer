import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN

class GeneralizedBlock(nn.Module):
    """ ConvNeXtV2 Block generalized to allow different input and output channels and different expansion ratios.
    
    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for MLP layers.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, in_dim, out_dim, expansion_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim) # depthwise conv
        self.norm = LayerNorm(in_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(in_dim, expansion_ratio * in_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(expansion_ratio * in_dim)
        self.pwconv2 = nn.Linear(expansion_ratio * in_dim, out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.residual_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = self.residual_proj(input) + self.drop_path(x)
        return x