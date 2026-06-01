# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .WaveMamba import *

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * nn.functional.sigmoid(x2)

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma

class SimpleBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2):
        super().__init__()
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = nn.GroupNorm(1, c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.conv1(self.norm1(x))
        x = self.sg(x)
        return inp + x * self.beta

#https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class RoPE2D(nn.Module):
    """Dynamically generates 2D Axial Rotary Position Embeddings."""
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.dim = head_dim // 2  # Split channels evenly between H and W
        
        # Precompute the inverse frequencies for 1D RoPE on each axis
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache matricies to avoid duplicate computation
        self.rope_cache = {}

    def _get_sin_cos(self, pos):
        # pos: (Length,) -> returns (Length, dim)
        sinusoid_inp = torch.outer(pos, self.inv_freq)
        emb = torch.cat((sinusoid_inp, sinusoid_inp), dim=-1)
        return emb.sin(), emb.cos()

    def forward(self, H, W, device):
        if (H, W) in self.rope_cache:
            return self.rope_cache[(H, W, str(device))]
        pos_h = torch.arange(H, device=device, dtype=torch.float32)
        pos_w = torch.arange(W, device=device, dtype=torch.float32)
        
        sin_h, cos_h = self._get_sin_cos(pos_h)
        sin_w, cos_w = self._get_sin_cos(pos_w)
        
        # Broadcast across the spatial grid
        sin_h = sin_h.unsqueeze(1).expand(-1, W, -1)
        cos_h = cos_h.unsqueeze(1).expand(-1, W, -1)
        sin_w = sin_w.unsqueeze(0).expand(H, -1, -1)
        cos_w = cos_w.unsqueeze(0).expand(H, -1, -1)
        
        # Flatten to sequence dimension (H*W, dim) and add batch/head dimensions
        sin_h = sin_h.reshape(-1, self.dim).unsqueeze(0).unsqueeze(0)
        cos_h = cos_h.reshape(-1, self.dim).unsqueeze(0).unsqueeze(0)
        sin_w = sin_w.reshape(-1, self.dim).unsqueeze(0).unsqueeze(0)
        cos_w = cos_w.reshape(-1, self.dim).unsqueeze(0).unsqueeze(0)

        # Cache results
        self.rope_cache[(H, W, str(device))] = ((sin_h, cos_h), (sin_w, cos_w))
        return (sin_h, cos_h), (sin_w, cos_w)


def rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)


def apply_rope_2d(x, rope_mats):
    # x shape: (B, num_heads, L, head_dim)
    (sin_h, cos_h), (sin_w, cos_w) = rope_mats
    dim = x.shape[-1] // 2
    
    # Process independent halves for H and W components
    x_h, x_w = x[..., :dim], x[..., dim:]
    
    x_h = (x_h * cos_h) + (rotate_half(x_h) * sin_h)
    x_w = (x_w * cos_w) + (rotate_half(x_w) * sin_w)
    
    return torch.cat([x_h, x_w], dim=-1)


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_mats):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply axial rotary embeddings to queries and keys
        q = apply_rope_2d(q, rope_mats)
        k = apply_rope_2d(k, rope_mats)
        
        # QK norm
        q, k = self.q_norm(q), self.k_norm(k)

        # Standard scaled dot-product attention
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        
        # out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, time_emb_dim=None):
        super().__init__()
        self.has_time = time_emb_dim is not None
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=(not self.has_time))
        self.attn = RoPEAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=(not self.has_time))
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        if self.has_time:
            # AdaLN-Zero Initialization setup
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 6 * dim, bias=True)
            )
            # Initialize to identity behavior at startup
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, rope_mats, t_emb=None):
        if self.has_time and t_emb is not None:
            # Chunk into 6 parameters (scale, shift, gate) for Attention and MLP paths
            mod = self.adaLN_modulation(t_emb).unsqueeze(1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
            
            # Attention with Adaptive LayerNorm modulation
            x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
            x = x + gate_msa * self.attn(x_norm, rope_mats)
            
            # MLP with Adaptive LayerNorm modulation
            x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = x + gate_mlp * self.mlp(x_norm)
        else:
            # Fallback to standard deterministic Transformer block
            x = x + self.attn(self.norm1(x), rope_mats)
            x = x + self.mlp(self.norm2(x))
            
        return x


class DiTBottleneck(nn.Module):
    """The main wrapper module to slot directly into your restoration bottleneck."""
    def __init__(self, dim, depth=4, num_heads=8, mlp_ratio=4.0, time_emb_dim=None):
        super().__init__()
        self.dim = dim
        head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert head_dim % 2 == 0, "head_dim must be even to safely apply split 2D RoPE"
        
        self.rope_gen = RoPE2D(head_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, time_emb_dim)
            for _ in range(depth)
        ])

    def forward(self, x, t_emb=None):
        """
        Args:
            x: Input feature tensor of shape (B, C, H, W) where C == dim
            t_emb: Optional time/conditioning tensor of shape (B, time_emb_dim)
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"Expected {self.dim} channels, but got {C}."
        
        # 1. Transition to sequence structure: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # 2. Build 2D Rotary Sine/Cosine grids matched to the current spatial window
        rope_mats = self.rope_gen(H, W, x.device)
        
        # 3. Process features through the DiT Stack
        for block in self.blocks:
            x_flat = block(x_flat, rope_mats, t_emb=t_emb)
            
        # 4. Restore original image block geometry: (B, H*W, C) -> (B, C, H, W)
        out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

class DemoNAFNetDITSigmoidModifiedSCA(nn.Module):

    def __init__(self, img_channel=3, in_channels=6, width=16, 
                 middle_blk_num=(0,1), enc_blk_nums=[], dec_blk_nums=[], mask=True, num_heads=4,
                 residual=False):
        super().__init__()
 
        # img_channel = in_channels // 2
        self.img_channel = img_channel
        if in_channels is None:
            in_channels = img_channel
        self.mask = mask
        self.residual = residual

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
 
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num, snum in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                    *[LFSSBlock(chan) for _ in range(snum)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num[0])],
                DiTBottleneck(dim=chan, depth=middle_blk_num[1], num_heads=num_heads, time_emb_dim=None)
            )

        for num, snum in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)],
                    *[LFSSBlock(chan) for _ in range(snum)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x[:, :, :H, :W]

        if self.mask:
            sparse, mask = inp.chunk(2, dim=1)
            x = x * (mask==0) + sparse

        if self.residual:
            _, c, _, _ = x.shape
            x = x + inp[:,:c]
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



