import torch.nn.functional as F
import torch
import torch.nn as nn


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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionedChannelAttention(nn.Module):
    def __init__(self, dims, cat_dims, out_dim=0):
        super().__init__()
        in_dim = dims + cat_dims
        if not out_dim:
            out_dim = dims
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca
    

class NKA(nn.Module):
    def __init__(self, dim, channel_reduction = 8):
        super().__init__()

        reduced_channels = dim // channel_reduction
        self.proj_1 = nn.Conv2d(dim, reduced_channels, 1, 1, 0)
        self.dwconv = nn.Conv2d(reduced_channels, reduced_channels, 3, 1, 1, groups=reduced_channels)
        self.proj_2 = nn.Conv2d(reduced_channels, reduced_channels * 2, 1, 1, 0)
        self.sg = SimpleGate()
        self.attention = nn.Conv2d(reduced_channels, dim, 1, 1, 0)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # First projection to a smaller dimension
        y = self.proj_1(x)
        # DW conv
        attn = self.dwconv(y)
        # PW to increase channel count for SG
        attn = self.proj_2(attn)
        # Non-linearity
        attn = self.sg(attn)
        # Back to original dimensions
        out = x * self.attention(attn)
        return out
    

class CondFuser(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan, out_dim=chan)
        self.sig = nn.Sigmoid()

        self.sa = nn.Sequential(
            nn.Conv2d(in_channels = 2 * chan, out_channels=1, kernel_size=3, 
                      padding=1, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x2 = 1 * self.sig(self.cca(x, cond)) * self.sa(x) * x2
        return x1 + x2

    
class CondFuserAdd(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()

    def forward(self, x1, x2, cond):
        return x1 + x2
    
class CondFuserV2(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)
        self.spa = NKA(chan * 2)

    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        spa = torch.sigmoid(self.spa(x)) * x
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

 
class CondFuserV3(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)
        self.spa = nn.Conv2d(
            in_channels=chan * 2,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        spa = torch.sigmoid(self.spa(x))

        x1, x2 = x.chunk(2, dim=1)
        return x1 * spa + x2 * (1 - spa)

class CondFuserV4(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)
        self.pw = nn.Conv2d(chan * 2, chan, 1, stride=1, padding=0, groups=1)
    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        x = self.pw(x)
        return x
    

class Branch(nn.Module):
    def __init__(self, c, kernel_size, cond_chans, expand=2 ):
            super().__init__()
            channels = c * expand
            self.norm = nn.GroupNorm(1, c)
            self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )
            if kernel_size > 0:
                self.conv2 = nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding="same",
                    stride=1,
                    groups=channels,
                    bias=True,
                )
            else:
                self.conv2 =  nn.Identity()

            # Simplified Channel Attention
            self.sca = ConditionedChannelAttention(channels // 2, cond_chans)

            self.conv3 = nn.Conv2d(
                in_channels=channels // 2,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            # SimpleGate
            self.sg = SimpleGate()
            self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x, cond = inp
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x, cond) * x
        x = self.conv3(x)

        return (inp[0] + self.alpha * x, cond)


class Tree(nn.Module):
    def __init__(self, c, kernels=[5,  3, 0], expand=2, cond_chans=0):
        super().__init__()
        self.trunk = nn.Sequential(*[Branch(c, k, cond_chans=cond_chans, expand=expand) for k in kernels])

    def forward(self, input):
        output = self.trunk(input)
        return output
    

import torch.nn.functional as F


class ConditioningCNN(nn.Module):
    def __init__(self, in_channels=4, num_logits=128):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_logits (int): The desired size of the output 1D logit vector.
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding='same'),
            nn.ReLU(inplace=True)
        )
        
        self.logit_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_logits)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.logit_head(x)
        return x
    
class Forest(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        chans = [],
        cond_input=1,
        cond_output=32,
        expand_dims=2,
        drop_out_rate=0.0,
        drop_out_rate_increment=0.0,
        rggb = False,
        use_CondFuserV2 = False,
        use_add = False,
        use_CondFuserV3 = False,
        use_CondFuserV4 = False,
        use_cond_net = False,
        cond_net_num = 32, 
        use_input_stats=False,
        **kwargs,
    ):
        super().__init__()
        width = chans[0]

        self.expand_dims = expand_dims
        self.conditioning_gen = nn.Sequential(
            nn.Linear(cond_input, 64), nn.ReLU(), nn.Dropout(drop_out_rate), nn.Linear(64, cond_output),
        )


        self.use_cond_net = use_cond_net
        if use_cond_net:
            self.cond_net = ConditioningCNN(in_channels=in_channels, num_logits=cond_net_num)
            cond_output = cond_output + cond_net_num

        self.use_input_stats = use_input_stats
        if use_input_stats:
            cond_output = cond_output + in_channels * 2

        self.rggb = rggb
        if not rggb:
            self.intro = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.intro = nn.Sequential(
                
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=width * 2 ** 2,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=1,
                    bias=True,
                ),
                nn.PixelShuffle(2)
            )
            
            nn.Conv2d(
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
        self.merges = nn.ModuleList()


        # for num in enc_blk_nums:
        for i in range(len(enc_blk_nums)):
            current_chan = chans[i]
            next_chan = chans[i + 1]
            num = enc_blk_nums[i]
            self.encoders.append(
                nn.Sequential(
                    *[
                        Tree(current_chan, cond_chans=cond_output)
                        for _ in range(num)
                    ]
                )
            )
            drop_out_rate += drop_out_rate_increment 
            self.downs.append(nn.Conv2d(current_chan, next_chan, 2, 2))

        self.middle_blks = nn.Sequential(
            *[
                Tree(next_chan, cond_chans=cond_output)
                for _ in range(middle_blk_num)
            ]
        )

        for i in range(len(dec_blk_nums)):
            current_chan = chans[-i-1]
            next_chan = chans[-i-2]
            num = dec_blk_nums[i]
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(current_chan, next_chan * 2 ** 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            drop_out_rate -= drop_out_rate_increment 
            if use_CondFuserV2:
                self.merges.append(CondFuserV2(next_chan, cond_chan=cond_output))
            elif use_add:
                self.merges.append(CondFuserAdd(next_chan, cond_chan=cond_output))
            elif use_CondFuserV3:
                self.merges.append(CondFuserV3(next_chan, cond_chan=cond_output))
            elif use_CondFuserV4:
                self.merges.append(CondFuserV4(next_chan, cond_chan=cond_output))
            else:
                self.merges.append(CondFuser(next_chan, cond_chan=cond_output))

            self.decoders.append(
                nn.Sequential(
                    *[
                        Tree(next_chan, cond_chans=cond_output)
                        for _ in range(num)
                    ]
                )
            )


        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond_in):
        # Conditioning:
        cond = self.conditioning_gen(cond_in)

        # if self.use_cond_net:
        #     extra_cond = self.cond_net(inp)
        #     cond = torch.cat([cond, extra_cond], dim=1)
        # if self.use_input_stats:
        #     mu = inp.mean((2,3), keepdim=True)
        #     var = (inp - mu).pow(2).mean((2,3), keepdim=False)
        #     mu = mu.squeeze(-1).squeeze(-1)
        #     cond = torch.cat([cond, mu, var], dim=1)
            
        B, C, H, W = inp.shape
        if self.rggb:
            H = 2 * H
            W = 2 * W
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder((x, cond))[0]
            encs.append(x)
            x = down(x)

        x = self.middle_blks((x, cond))[0]

        for decoder, up, merge, enc_skip in zip(self.decoders, self.ups, self.merges, encs[::-1]):
            x = up(x)
            x = merge(x, enc_skip, cond)
            x = decoder((x, cond))[0]

        x = self.ending(x)
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class ModelWrapper(nn.Module):
    def __init__(self, **kwargs):
        self.gamma = 1
        if 'gamma' in kwargs:
            self.gamma = kwargs.pop('gamma')
        super().__init__()
        self.model = Restorer(
            **kwargs
        )

    def forward(self, x, cond, residual):
        x = x.clip(0, 1) ** (1. / self.gamma)
        residual = residual.clip(0, 1) ** (1. / self.gamma)
        output = self.model(x, cond)
        output = (residual + output).clip(0, 1) ** (self.gamma)
        return output
    

def make_full_model_RGGB(params, model_name=None):
    model = ModelWrapper(**params)
    if not model_name is None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model



from src.Restorer.Cond_NAF_demosaic import DemosaicingFromRGGB

class ModelWrapperNoRes(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if 'gamma' in kwargs:
            kwargs.pop('gamma')

        self.demosaicer = DemosaicingFromRGGB()
        self.model = Restorer(
            **kwargs
        )

    def forward(self, rggb, cond, *args):
        debayered = self.demosaicer(rggb, cond)
        output = self.model(rggb, cond)
        output = (debayered + output)
        return output
    

def make_full_model_RGGB_NoRes(params, model_name=None):
    model = ModelWrapperNoRes(**params)
    if not model_name is None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model


class ModelWrapperPS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if 'gamma' in kwargs:
            kwargs.pop('gamma')
        kwargs['rggb'] = False
        kwargs['out_channels'] = kwargs['out_channels'] * 4
        self.demosaicer = DemosaicingFromRGGB()
        self.model = Restorer(
            **kwargs
        )
        self.ps = nn.PixelShuffle(2)

    def forward(self, rggb, cond, *args):
        debayered = self.demosaicer(rggb, cond)
        output = self.model(rggb, cond)
        output = self.ps(output)
        output = (debayered + output)
        return output
    

def make_full_model_PS(params, model_name=None):
    model = ModelWrapperPS(**params)
    if not model_name is None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model



class ModelWrapperDemoIn(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if 'gamma' in kwargs:
            kwargs.pop('gamma')

        self.demosaicer = DemosaicingFromRGGB()
        self.model = Forest(
            **kwargs
        )

    def forward(self, rggb, cond, *args):
        debayered = self.demosaicer(rggb, cond)
        output = self.model(debayered, cond)
        output = (debayered + output)
        return output
    

def make_full_model_RGGB_DemoIn(params, model_name=None):
    model = ModelWrapperDemoIn(**params)
    if not model_name is None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model
