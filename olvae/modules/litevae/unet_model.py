# ADM Style U-Net without spatial downsampling - up/down only applies in the channel dim
# Model is configured to support arbitrary resnet block count

import math
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

# simple down and up sample layer - don't change spatially, but increase the hidden dim
def get_downsample_layer(in_dim, hidden_dim, is_last):
    return nn.Conv2d(in_dim, hidden_dim, 1, padding=0)

def get_upsample_layer(in_dim, hidden_dim, is_last):
    return nn.Conv2d(in_dim, hidden_dim, 1, padding=0)

class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=8
                ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # project the residual if the dims don't match
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels=out_channels,
            kernel_size=1) if in_channels != out_channels else nn.Identity()

        # primary compute elements
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()

    def forward(self, x):

        # residual skip / project to adapt hidden dim
        residual = self.residual_conv(x)

        # first conv-norm-act
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        # second conv-norm-act
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # add the residual
        return x + residual


class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 out_channels=None,
                 num_resblocks=2,
                 channel_mult=[1, 2, 4],
                 image_size=64
                ):
        super(UNet, self).__init__()

        # precompute the hidden dims, out_channels and save the states
        hidden_dims = [m*channels for m in channel_mult]

        out_channels = in_channels if out_channels is None else out_channels
            
        self.sample_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        
        # project in convolution
        self.init_conv = nn.Conv2d(in_channels,
                                   out_channels=hidden_dims[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        ####### The U-Net #############
        down_blocks = []
        in_dim = hidden_dims[0]

        # Down blocks
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            down_blocks.append(
                nn.ModuleList([
                    *[ResidualBlock(in_dim, in_dim) for _ in range(num_resblocks)],
                    get_downsample_layer(in_dim, hidden_dim, is_last)
                ]))
            in_dim = hidden_dim

        self.down_blocks = nn.ModuleList(down_blocks)
        # end down blocks

        # middle blocks
        mid_dim = hidden_dims[-1]
        self.mid_block = nn.ModuleList([
                    ResidualBlock(mid_dim, mid_dim) for _ in range(num_resblocks)
                ])

        # up blocks
        up_blocks = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            up_blocks.append(
                nn.ModuleList([
                    get_upsample_layer(in_dim, hidden_dim, is_last),
                    *[ResidualBlock(2*hidden_dim, hidden_dim) for _ in range(num_resblocks)]
                    
                ]))
            in_dim = hidden_dim

        self.up_blocks = nn.ModuleList(up_blocks)
        # end up blocks

        # project out convolution
        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        # project input
        x = self.init_conv(x)

        # down blocks
        skips = []
        for l, level in enumerate(self.down_blocks):
            for i, block in enumerate(level):
                x = block(x)
                if i != len(level)-1:
                    skips.append(x)
        # end down
        
        # middle blocks
        for i, block in enumerate(self.mid_block):
            x = block(x)

        # up blocks
        for l, level in enumerate(self.up_blocks):
            for i, block in enumerate(level):
                if i != 0:
                    hskip = skips.pop()
                    h = torch.cat((x, hskip), dim=1)
                else:
                    h = x
                x = block(h)
        # end up blocks

        # output projection
        out = self.conv_out(x)
        return out