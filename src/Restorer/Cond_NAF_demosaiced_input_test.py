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
    def __init__(self, dims, cat_dims):
        super().__init__()
        in_dim = dims + cat_dims
        self.mlp = nn.Sequential(nn.Linear(in_dim, dims))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, conditioning):
        pool = self.pool(x)
        conditioning = conditioning.unsqueeze(-1).unsqueeze(-1)
        cat_channels = torch.cat([pool, conditioning], dim=1)
        cat_channels = cat_channels.permute(0, 2, 3, 1)
        ca = self.mlp(cat_channels).permute(0, 3, 1, 2)

        return ca
    
class CondFuser(nn.Module):
    def __init__(self, chan, cond_chan=1):
        super().__init__()
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)
        # self.spa = nn.Conv2d(
        #     in_channels=chan * 2,
        #     out_channels=1,
        #     kernel_size=3,
        #     padding=1,
        #     stride=1,
        #     groups=1,
        #     bias=True,
        # )

    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        # spa = torch.sigmoid(self.spa(x))

        x1, x2 = x.chunk(2, dim=1)
        # return x1 * spa + x2 * (1 - spa)
        return x1 + x2


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
        self.cca = ConditionedChannelAttention(chan * 2, cond_chan)

    def forward(self, x1, x2, cond):
        x = torch.cat([x1, x2], dim=1)
        x = self.cca(x, cond) * x
        x1, x2 = x.chunk(2, dim=1)
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
        self.sca_out = ConditionedChannelAttention(c, cond_chans)

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
        self.delta = nn.Parameter(torch.zeros((1, 1, 1, 1)), requires_grad=True)

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


        xp = (1 + self.delta * self.sca_out(x, cond)) * x


        return (y + x * self.gamma, cond)
    

class NAFBlock0_learned_norm(nn.Module):
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
        self.sca_mul = ConditionedChannelAttention(c, cond_chans)
        self.sca_add = ConditionedChannelAttention(c, cond_chans)

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
        normed = self.norm2(y)

        # Input mediated channel attention, obstensibly to mitigate the effects of group norm on flat scenes
        x = (1 + self.sca_mul(inp, cond)) * normed + self.sca_add(inp, cond)

        x = self.conv4(x)
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)

        x = self.dropout2(x)



        return (y + x * self.gamma, cond)
    
class NAFBlock0AdjustedNorm(nn.Module):
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

        self.norm1 = LayerNorm2dAdjusted(c)
        self.norm2 = LayerNorm2dAdjusted(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.sca_mul = ConditionedChannelAttention(c, cond_chans)
        self.sca_add = ConditionedChannelAttention(c, cond_chans)

    def forward(self, input):
        inp = input[0]
        cond = input[1]

        x = inp

        x = self.norm1(x, mu, var)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x, cond)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # Channel Mixing
        normed = self.norm2(y, mu, var)

        # Input mediated channel attention, obstensibly to mitigate the effects of group norm on flat scenes
        # x = (1 + self.sca_mul(inp, cond)) * normed + self.sca_add(inp, cond)

        x = self.conv4(normed)
        x = self.sg(x)
        # x = self.grn(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return (y + x * self.gamma, cond, mu, var)


import torch.nn.functional as F

class SwiGLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Conv2d(input_dim, hidden_dim, 1, 1, 0, 1)
        self.w2 = nn.Conv2d(input_dim, hidden_dim, 1, 1, 0, 1)
        self.w3 = nn.Conv2d(hidden_dim, input_dim, 1, 1, 0, 1)
        
    def forward(self, x):
        gate = F.silu(self.w1(x)) 
        value = self.w2(x)
        x = gate * value 
        
        x = self.w3(x)
        return x
    
class AttnBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.0, cond_chans=0):
        super().__init__()
        
        self.dw = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=c,
            bias=True,
        )
        self.nka = NKA(c)

        self.sca = ConditionedChannelAttention(c, cond_chans)

        self.norm = nn.GroupNorm(1, c)
        
        self.swiglu = SwiGLU(c, int(c *  FFN_Expand))
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))


    def forward(self, input):
        inp = input[0]
        cond = input[1]

        x = self.dw(inp)
        x = self.nka(x)
        x = self.sca(x, cond) * x
        y = self.norm(inp + self.alpha * x )


        x = self.swiglu(y)
        x = y + self.beta * x
        return (x, cond)


class CondSEBlock(nn.Module):
    def __init__(self, chan, reduction=16, cond_chan=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(chan + cond_chan, chan // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(chan // reduction, chan, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, conditioning):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat([y, conditioning], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    


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
    
class Restorer(nn.Module):
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
        use_attnblock = False,
        use_NAFBlock0_learned_norm=False,
        use_cond_net = False,
        cond_net_num = 32, 
        use_input_stats=False,
        use_NAFBlock0AdjustedNorm=False,
    ):
        super().__init__()
        if use_attnblock:
            block = AttnBlock
        elif use_NAFBlock0_learned_norm:
            block = NAFBlock0_learned_norm
        elif use_NAFBlock0AdjustedNorm:
            block = NAFBlock0AdjustedNorm
        else:
            block = NAFBlock0

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
                        block(current_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )
            drop_out_rate += drop_out_rate_increment 
            self.downs.append(nn.Conv2d(current_chan, next_chan, 2, 2))

        self.middle_blks = nn.Sequential(
            *[
                block(next_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
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
                        block(next_chan, cond_chans=cond_output, drop_out_rate=drop_out_rate)
                        for _ in range(num)
                    ]
                )
            )

        self.base_merge = CondFuser(width, cond_chan=cond_output)
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
        base = x
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
        x = self.base_merge(x, base, cond)
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



class ModelWrapperDemoInput(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=(29) * 2 ** 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            ),
            nn.PixelShuffle(2)
        )

        if 'gamma' in kwargs:
            kwargs.pop('gamma')
        kwargs.pop('rggb')
        kwargs.pop('in_channels')

        self.demosaicer = DemosaicingFromRGGB()
        self.model = Restorer(
            rggb=False,
            in_channels = 32,
            **kwargs
        )


    def forward(self, rggb, cond, *args):
        debayered = self.demosaicer(rggb, cond)
        intro = self.intro(rggb)
        intro = torch.cat([debayered, intro], dim=-3)
        output = self.model(intro, cond)
        output = (debayered + output)
        return output
    

def make_full_model_RGGB_Debayer_Inp(params, model_name=None):
    model = ModelWrapperDemoInput(**params)
    if not model_name is None:
        state_dict = torch.load(model_name, map_location="cpu")
        model.load_state_dict(state_dict)
    return model
