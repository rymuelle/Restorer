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
import torch.nn.functional as tF

class LayerNorm2dAdjusted(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x, target_mu, target_var):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / torch.sqrt(var + self.eps)

        y = y * torch.sqrt(target_var + self.eps) + target_mu

        weight_view = self.weight.view(1, self.weight.size(0), 1, 1)
        bias_view = self.bias.view(1, self.bias.size(0), 1, 1)

        y = weight_view * y + bias_view
        return y

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / torch.sqrt(var + self.eps)

        weight_view = self.weight.view(1, self.weight.size(0), 1, 1)
        bias_view = self.bias.view(1, self.bias.size(0), 1, 1)

        y = weight_view * y + bias_view
        return y

class GlobalReasoningModule(nn.Module):
    def __init__(self, chans, global_chans):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=chans+global_chans, out_channels=chans+global_chans, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=chans+global_chans, out_channels=chans+global_chans, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)  
        self.split = [chans, global_chans]

    def forward(self, x, global_x):
        x = self.pool(x)
        x = torch.cat([x, global_x], dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x, global_x = torch.split(x, self.split, dim=-3)
        return x, global_x

class CondFuser(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels = 2 * chan, out_channels=chan, kernel_size=3, 
                      padding=1, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=2 * chan, out_channels=chan, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x2 = 1 * self.ca(x) * self.sa(x) * x2
        return x1 + x2

class CondFuserAdd(nn.Module):
    def __init__(self, chan):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, global_c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.grm = GlobalReasoningModule(c, global_c)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.delta = nn.Parameter(torch.zeros(1, global_c, 1, 1))
    def forward(self, inps):
        inp, global_inp = inps
        x = inp
        g_x = global_inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        attn, x_g = self.grm(x, g_x)
        x = x * attn
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return (y + x * self.gamma, global_inp + x_g * self.delta)


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                use_add = False, global_channel=16):
        super().__init__()
        self.global_channel = global_channel
        self.global_channel_producer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, global_channel, 1)
        )
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.merges = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, global_channel) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, global_channel) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, global_channel) for _ in range(num)]
                )
            )

            if use_add:
                self.merges.append(CondFuserAdd(chan))
            else:
                self.merges.append(CondFuser(chan))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        global_info = self.global_channel_producer(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, global_info = encoder((x, global_info))
            encs.append(x)
            x = down(x)

        x, global_info = self.middle_blks((x, global_info))

        for decoder, up, merge, enc_skip in zip(self.decoders, self.ups, self.merges, encs[::-1]):
            x = up(x)
            x = merge(x, enc_skip)
            x, global_info = decoder((x, global_info))

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = tF.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x