import torch
import torch.nn as nn
import torch.nn.functional as F

# Modulated Convolution from https://rdrr.io/github/rdinnager/styleganr/src/R/networks.R
def modulated_conv2d(
      x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
      w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
      s,                  # Style tensor: [batch_size, in_channels]
      demodulate  = True, # Apply weight demodulation?
      padding     = 0,    # Padding: int or [padH, padW]
      input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
      eps = 1e-8
    ):

    batch_size = x.shape[0]
    in_channels = x.shape[1]
    
    # Pre-normalize inputs.
    if demodulate:
        # Normalize 'w'
        # Calculate the mean of the squares of 'w' across dimensions I,K,K for the weight matrix (dims 1,2,3)
        w_scale = torch.mean(w ** 2, dim=(1, 2, 3), keepdim=True).rsqrt()  # Compute the reciprocal square root
        w = w * w_scale  # Scale 'w' by the computed normalization factor
        
        # Normalize 's'
        s_scale = torch.mean(s ** 2).rsqrt()  # Compute the reciprocal square root of the mean of the squares
        s = s * s_scale  # Scale 's' by the computed normalization factor
    
    # Modulate weights.
    w = w * s.view(1, -1, 1, 1) # (O,I,K,K)

    if demodulate:
        # Calculate demodulation coefficients:
        # Sum the squares of 'w' across dimensions I,K,K (dims 1,2,3)
        dcoefs = (w.pow(2).sum(dim=(1, 2, 3), keepdim=True) + eps).rsqrt() # keep dims leaves it as (O,I,K,K)
    
        # Demodulate weights:
        w = w * dcoefs

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(in_channels) # (I)
        # view in the same shape as the weights
        input_gain = input_gain.view(1, -1, 1, 1) # (O,I,K,K)
        # scale the weights
        w = w*input_gain

    # now apply the convolution operation
    x = F.conv2d(x, w, padding=padding, 
             groups=1, bias=None, 
             stride=1)

    return x

class SMC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels = None,
        kernel_size = 3,
        stride = None,
        padding = None,
        bias = True
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        padding_ = int(kernel_size // 2) if padding is None else padding
        stride_ = 1 if stride is None else stride
        self.padding = padding_
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            padding = padding_,
            stride = stride_,
            bias = bias
        )
        self.gain = nn.Parameter(torch.ones(1))
        self.scales = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        
        out = modulated_conv2d(
            x = x,
            w = self.conv.weight,
            s = self.scales,
            padding = self.padding,
            input_gain = self.gain,
        )
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1,-1, 1, 1)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels=None, 
                 conv_shortcut=False,
                 dropout=0.0
                ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = SMC(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = SMC(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    # end __init__

    def forward(self, x):
        h = x
        
        # 1st conv-act block
        h = self.conv1(h)
        h = F.silu(h) # nonlinearity

        # dropout
        h = self.dropout(h)

        # 2nd conv-act block
        h = self.conv2(h)
        h = F.silu(h) # nonlinearity

        # handle the shortcut scaling
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class Upsample(nn.Module):
    def __init__(self, 
                 in_channels, 
                 with_conv=True
                ):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsample
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # smooth
        if self.with_conv:
            x = self.conv(x)
        return x

class LiteVAE_Decoder(nn.Module):
    def __init__(self,
                 channels,
                 z_channels,
                 out_channels, 
                 channel_mult=(1,2,4,8), 
                 num_res_blocks=2,
                 dropout=0.0,
                 resamp_with_conv=True,
                ):
        super().__init__()
        # save states
        self.channels = channels
        self.num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.out_channels = out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(channel_mult)
        block_in = channels*channel_mult[self.num_resolutions-1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle block stack
        self.mid = nn.ModuleList([ ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout) for _ in range(num_res_blocks+1)
                                 ])

        # up-sample decoder
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = channels*channel_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            # end repeated blocks

            # upsampling
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # end 
        
        # final block - SMC includes normalization
        self.conv_norm = SMC(
            block_in,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv_out = nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, z):

        # z to block_in
        h = self.conv_in(z)

        # middle
        for block in self.mid:
            h = block(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end

        # output stack
        h = self.conv_norm(h)
        h = F.silu(h) # nonlinearity
        h = self.conv_out(h)
        return h