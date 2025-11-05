# ------------------------------------------------------------------------
# Modified from CGNet
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from Restorer.utils import (
    SimpleGate,
    ConditionedChannelAttention,
    LayerNorm2d,
    Block,
    ViTBlock,
)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=0, stide=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(
            nin,
            nin,
            kernel_size=kernel_size,
            stride=stide,
            padding=padding,
            groups=nin,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UpsampleWithFlops(nn.Upsample):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
        self.__flops__ = 0

    def forward(self, input):
        self.__flops__ += input.numel()
        return super(UpsampleWithFlops, self).forward(input)


class GlobalContextExtractor(nn.Module):
    def __init__(
        self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False
    ):
        super(GlobalContextExtractor, self).__init__()

        self.depthwise_separable_convs = nn.ModuleList(
            [
                depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
                for kernel_size, stride in zip(kernel_sizes, strides)
            ]
        )

    def forward(self, x):
        outputs = []
        for conv in self.depthwise_separable_convs:
            x = F.gelu(conv(x))
            outputs.append(x)
        return outputs


class CascadedGazeBlock(nn.Module):
    def __init__(
        self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0, cond_chans=0
    ):
        super().__init__()
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=self.dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.dw_channel,
            out_channels=self.dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=self.dw_channel,
            bias=True,
        )

        if self.GCE_Conv == 3:
            self.GCE = GlobalContextExtractor(
                c=c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4]
            )

            self.project_out = nn.Conv2d(int(self.dw_channel * 2.5), c, kernel_size=1)

            self.sca = ConditionedChannelAttention(
                int(self.dw_channel * 2.5), cond_chans
            )

        else:
            self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3], strides=[2, 3])

            self.project_out = nn.Conv2d(self.dw_channel * 2, c, kernel_size=1)

            self.sca = ConditionedChannelAttention(int(self.dw_channel * 2), cond_chans)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # self.grn = GRN(ffn_channel // 2)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, input):
        inp = input[0]
        cond = input[1]
        x = inp
        b, c, h, w = x.shape
        # # Nearest neighbor upsampling as part of the range fusion process
        self.upsample = UpsampleWithFlops(size=(h, w), mode="nearest")

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        # Global Context Extractor + Range fusion
        x_1, x_2 = x.chunk(2, dim=1)
        if self.GCE_Conv == 3:
            x1, x2, x3 = self.GCE(x_1 + x_2)
            x = torch.cat(
                [x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], dim=1
            )
        else:
            x1, x2 = self.GCE(x_1 + x_2)
            x = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim=1)
        x = self.sca(x, cond) * x
        x = self.project_out(x)

        x = self.dropout1(x)
        # channel-mixing
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return (y + x * self.gamma, cond)


class NAFBlock0(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, cond_chans=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = ConditionedChannelAttention(dw_channel // 2, cond_chans)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # self.grn = GRN(ffn_channel // 2)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, input):
        inp = input[0]
        cond = input[1]

        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x, cond)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # Channel Mixing
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return (y + x * self.gamma, cond)


class Restorer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        vit_blk_nums=[],
        dec_blk_nums=[],
        cond_input=1,
        cond_output=32,
        expand_dims=2,
        drop_path=0.0,
        drop_path_increment=0.0,
        dropout_rate=0.3
    ):
        super().__init__()

        self.expand_dims = expand_dims
        # self.film_gen =  FiLMGenerator(1, (width, width*2**len(enc_blk_nums), width, out_channels))
        # self.film_block = FiLMBlock()
        self.conditioning_gen = nn.Sequential(
            nn.Linear(cond_input, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, cond_output),
        )

        self.intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            num = enc_blk_nums[i]
            vit_num = vit_blk_nums[i]
            self.encoders.append(
                nn.Sequential(
                    *[
                        Block(chan, cond_chans=cond_output, expand_dim=self.expand_dims, drop_path=drop_path)
                        for _ in range(num)
                    ],
                    *[
                        ViTBlock(
                            chan, cond_chans=cond_output, expand_dim=self.expand_dims, drop_path=drop_path
                        )
                        for _ in range(vit_num)
                    ],
                )
            )
            drop_path += drop_path_increment 
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[
                Block(chan, cond_chans=cond_output, expand_dim=self.expand_dims, drop_path=drop_path)
                for _ in range(middle_blk_num)
            ]
        )

        for i in range(len(dec_blk_nums)):
            num = dec_blk_nums[i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            drop_path -= drop_path_increment 
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        Block(chan, cond_chans=cond_output, expand_dim=self.expand_dims, drop_path=drop_path)
                        for _ in range(num)
                    ]
                )
            )


        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond_in):
        # Conditioning:
        cond = self.conditioning_gen(cond_in)

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder((x, cond))[0]
            encs.append(x)
            x = down(x)

        x = self.middle_blks((x, cond))[0]

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder((x, cond))[0]

        x = self.ending(x)
        # x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class AddPixelShuffleWithPassThrough(nn.Module):
    def __init__(self, model, in_channels=4, out_channels=3):
        super().__init__()
        self.model = model
        self.ps = nn.PixelShuffle(2)
        self.upscale = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x, iso):
        inp = x
        x = self.model(x, iso)
        x = self.ps(x)
        res = self.upscale(inp)
        return x + res


class AddPixelShuffle(nn.Module):
    def __init__(self, model, in_channels=4, out_channels=3):
        super().__init__()
        self.model = model
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, iso):
        x = self.model(x, iso)
        x = self.ps(x)
        return x




class NAFBlockCond(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, cond_chans=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = ConditionedChannelAttention(dw_channel // 2, cond_chans)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # self.grn = GRN(ffn_channel // 2)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, cond):

        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x, cond)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # Channel Mixing
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma, cond
