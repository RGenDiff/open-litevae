import math

import numpy as np
import torch
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from abc import abstractmethod

from attention import SpatialTransformer

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    #for p in module.parameters():
    #    p.detach().zero_()
    return module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
class TimestepCombinedBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """      

class TimestepEmbedSequential(nn.Sequential, TimestepBlock,TimestepCombinedBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TimestepCombinedBlock):
                x = layer(x, emb, context)
            else:
                x = layer(x)
        return x    


    
    
def Conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

def Linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

BatchNorm2d = nn.BatchNorm2d

class _AdaptiveConv(TimestepBlock):
    """
    An adaptive kernel module with K filter banks
    :param channels: channels in the inputs and outputs.
    :param emb_size: the embdding size for modulation
    :param banks: the number of filter banks
    :param out_channels: the number of output channels from the convolution

    """

    def __init__(self, channels, banks=1, emb_channels=None, out_channels=None, 
                 bias=True, demod=True, eps=1.e-6, kernel=3, stride=1, padding=1):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.banks = banks
        self.out_channels = default(out_channels,channels)
        self.bias = bias
        self.demod = demod
        self.eps = eps
        self.stride = stride
        self.padding = padding

        if emb_channels is not None:
            lin = nn.Linear(emb_channels, 2*channels, bias=True)
            nn.init.uniform_(lin.weight, -0.01, 0.01)
            nn.init.constant_(lin.bias, 0.0)

            self.modulation = nn.Sequential(
                act(),
                nn.utils.spectral_norm(lin)
            ) 

        else:
            self.modulation = None

        self.weight = nn.Parameter(torch.randn(banks, self.out_channels, channels, kernel, kernel))
        nn.init.kaiming_normal_(self.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')


        if bias:
            self.filterbias = nn.Parameter(torch.randn(banks,self.out_channels))
            nn.init.constant_(self.filterbias, 0.0)



    def forward(self, x, emb):

        b, c, h, w = x.shape

        # selectively apply the bank selection
        tweight = self.weight.repeat(x.size(0),1,1,1,1)
        tbias = self.filterbias.repeat(x.size(0),1) if self.bias else None

        if self.emb_channels is not None:
            shift, scale = self.modulation(emb).chunk(2, dim=1)
            # rearrange the shift/scale shape to match the filter shape
            scale = rearrange(scale, 'b i -> b 1 i 1 1')
            shift = rearrange(shift, 'b i -> b 1 i 1 1')
            # perform the modulation
            weights = tweight*(scale + 1) + shift
        else:
            weights = tweight

        # get the modulation
        if self.demod:
            inv_norm = reduce(weights**2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights*inv_norm


        x = rearrange(x, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        if self.bias:
            tbias = rearrange(tbias, 'b o ... -> (b o) ...')

        x = F.conv2d(x, weights, padding=self.padding, 
                     groups=b, bias=tbias if self.bias else None, 
                     stride=self.stride)

        return rearrange(x, '1 (b o) ... -> b o ...', b=b)

# Wrap with spectral norm
class AdaptiveConv(TimestepBlock):
    def __init__(self, *args, **kwargs):
        super().__init__()  # Add necessary arguments for TimestepBlock
        self.conv = nn.utils.spectral_norm(_AdaptiveConv(*args, **kwargs))

    def forward(self, x, emb):
        return self.conv(x.contiguous(), emb.contiguous())

