from ConvNeXtUNet.convnextv2 import Block 
from ConvNeXtUNet.convnextv2_style_blocks import GeneralizedBlock

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from ConvNeXtUNet.utils import LayerNorm

class ConvNeXtV2Encoder(nn.Module):
    """ ConvNeXt V2 Style Encoder/Backbone
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(self, in_chans=3, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_values = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            skip_values.append(x)
        return skip_values[::-1]
    

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class ConvNeXtV2Decoder(nn.Module):
    """ ConvNeXt V2 Style Decoder
        
    Args:
        dims (tuple(int)): Number of channels of the skip connection at each stage of the encoder in reverse order.
        out_dim (int): Final output channels. Default: 1
        depths (tuple(int)): Number of blocks at each stage. Default: [2, 2, 2]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        use_pixel_shuffle (bool): Use pixel shuffle for the last upsampling layer. Default: False
        concat_original_image (bool): Concat the original image onto the output and run two more conv blocks. Default: False
        ms_output(bool): Output multiscale outputs for training. Default: False
    """
    def __init__(self, dims=[320, 160, 80, 40], out_dim=1, 
                 depths=[2, 2, 2], drop_path_rate=0.,
                 use_pixel_shuffle=False,
                 concat_original_image=False,
                 ms_output=False):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.ms_output = ms_output

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for idx in range(len(dims)-1):
            upconv = nn.Sequential(
                Interpolate(scale_factor=2),
                GeneralizedBlock(dims[idx], dims[idx+1]),
                LayerNorm(dims[idx+1], eps=1e-6, data_format="channels_first")
            )
            self.upconvs.append(upconv)


            decoder_block_list = []
            for jdx in range(depths[idx]):
                # Reduce the dimensions on the last block
                _in_dim = 2 * dims[idx+1] if jdx == 0 else dims[idx+1]
                decoder_block_list.append(GeneralizedBlock(_in_dim, dims[idx+1], drop_path=dp_rates[cur + jdx]))

            decoder_block =  nn.Sequential(
                *decoder_block_list
            )
            self.decoder_blocks.append(decoder_block)
            cur += depths[idx]

        if self.ms_output:
            self.ms_convs = nn.ModuleList()
            for idx in range(len(dims)):
                ms_conv = nn.Sequential(
                    nn.Conv2d(dims[idx], out_dim, kernel_size=3, padding=1),
                )
                self.ms_convs.append(ms_conv) 

        if use_pixel_shuffle:
            assert dims[-1] / 16 == dims[-1] // 16, f'Input to pixel shuffle must be a multiple of 16, {dims[-1]} dims found.'
            self.root = nn.Sequential(
                nn.PixelShuffle(4),
                GeneralizedBlock(dims[-1]//16, out_dim),
            )    
        else:
            self.root = nn.Sequential(
                Interpolate(scale_factor=4),
                GeneralizedBlock(dims[-1], out_dim),
            )

        self.concat_original_image = concat_original_image
        if concat_original_image:
            self.image_concat_blocks = nn.Sequential(
                GeneralizedBlock(2 * out_dim, out_dim),
                GeneralizedBlock(out_dim, out_dim),
            )
  
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, input, encoder_features):
        x = encoder_features[0]

        if self.ms_output:
            ms_output = []
            ms_output.append(self.ms_convs[0](x))

        for idx in range(len(encoder_features)-1):
            x = self.upconvs[idx](x)
            skip = encoder_features[idx+1]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx](x)
            if self.ms_output:
                ms_out = self.ms_convs[idx+1](x)
                ms_output.append(ms_out)
                
        x = self.root(x)
        if self.concat_original_image:
            x = torch.cat([input, x], dim=1)
            x = self.image_concat_blocks(x)

        if self.ms_output:
            ms_output.append(x)
            return ms_output
        return x
 
class ConvNeXtV2Unet(nn.Module):
    """ ConvNeXt V2 Style Decoder
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        enc_depths (tuple(int)): Number of blocks at each stage for the decoder. Default: [2, 2, 6, 2]
        dims (int): Feature dimension at each stage. Default: [40, 80, 160, 320]
        dec_depths (tuple(int)): Number of blocks at each stage. Default: [2, 6, 2]
        out_dim (int): Final output channels. Default: 3
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        use_pixel_shuffle (bool): Use pixel shuffle for the last upsampling layer. Default: False
        concat_original_image (bool): Concat the original image onto the output and run two more conv blocks. Default: False
        ms_output(bool): Output multiscale outputs for training. Default: False
    """
    def __init__(self, in_chans=3, 
                 enc_depths=[2, 2, 6, 2], 
                 dims=[40, 80, 160, 320],
                 dec_depths=[2, 6, 2], out_dim=3, 
                 drop_path_rate=0.,
                 use_pixel_shuffle=False,
                 concat_original_image=False,
                 ms_output=False
                    ):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(in_chans=in_chans, 
                                         depths=enc_depths, 
                                         dims=dims, 
                                         drop_path_rate=drop_path_rate
                                         )
        
        # The dims need to go in reverse order as we feed depth first
        self.decoder = ConvNeXtV2Decoder(depths=dec_depths,
                                         dims=dims[::-1],
                                         out_dim=out_dim,
                                         drop_path_rate=drop_path_rate,
                                         use_pixel_shuffle=use_pixel_shuffle,
                                         concat_original_image=concat_original_image,
                                         ms_output=ms_output
                                         )
        
    def set_ms_output(self, ms_output):
        self.decoder = ms_output

    def forward(self, x):
        skip_connections = self.encoder(x)
        output = self.decoder(x, skip_connections)
        return output
    

def convnextv2unet_atto(**kwargs):
    model = ConvNeXtV2Unet(in_chans=3, 
                 enc_depths=[2, 2, 6, 2], 
                 dims=[40, 80, 160, 320],
                 dec_depths=[2, 6, 2], 
                 **kwargs
                 )
    return model

def convnextv2unet_pico(**kwargs):
    model = ConvNeXtV2Unet(in_chans=3, 
                 enc_depths=[2, 2, 6, 2], 
                 dims=[64, 128, 256, 512],
                 dec_depths=[2, 6, 2],
                 **kwargs
                 )
    return model

def convnextv2unet_base(**kwargs):
    model = ConvNeXtV2Unet(in_chans=3, 
                 enc_depths=[3, 3, 27, 3], 
                 dims=[128, 256, 512, 1024],
                 dec_depths=[3, 7, 3],
                 **kwargs
                 )
    return model