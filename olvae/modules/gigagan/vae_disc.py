import math

import numpy as np
import torch

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

from attention import SpatialTransformer
from layers import TimestepEmbedSequential, BatchNorm2d, AdaptiveConv, Conv2d


_USE_SPECTRAL = True
act = nn.LeakyReLU



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('AdaptiveConv') != -1:
        pass
    elif classname.find('MultiConv') != -1:
        pass
    elif classname.find('Conv') != -1:
        if _USE_SPECTRAL:
            pass
        else:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        except Exception:
            pass
            

class SimplePredictor(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, n_layers=3, use_actnorm=False, embedding_channels=512, use_bn=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(SimplePredictor, self).__init__()
        if not use_actnorm:
            norm_layer = BatchNorm2d
        else:
            norm_layer = ActNorm
        use_bias = norm_layer != BatchNorm2d

       
        # first layer
        sequence = [
            AdaptiveConv(input_nc, out_channels=input_nc, banks=1, emb_channels=embedding_channels, 
         bias=True, demod=False, kernel=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            sequence += [
                AdaptiveConv(input_nc, out_channels=input_nc, banks=1, emb_channels=embedding_channels, 
         bias=use_bias, demod=False, kernel=1, stride=1, padding=0),
                norm_layer(input_nc) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=False)
            ]

        sequence += [
            Conv2d(input_nc, 1, kernel_size=1, stride=1, padding=0)
        
        ]  # output 1 channel prediction map
        self.main = TimestepEmbedSequential(*sequence)

    def forward(self, input, emb=None):
        """Standard forward."""
        return self.main(input, emb)

        
class MultiConv(nn.Module):
    def __init__(
        self,
        img_channels,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    ):
        super().__init__()
        
        self.img_channels = img_channels
        if in_channels is None:
            in_channels = out_channels
        
        # check for the special case where img_channels = None, then don't create
        self.img_conv = nn.Sequential(
                        Conv2d(in_channels=img_channels, 
                          out_channels=in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True,
                              ),  
                         nn.LeakyReLU(0.1, inplace=False),
                       ) if img_channels is not None else None
        
        # check for the special case where in_channels is None, then don't create
        self.chan_conv = Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias,
                         )
        
    def forward(self, x):
        b,c,_,_ = x.shape
        
        if c == self.img_channels:
            x = self.img_conv(x)
            
        return self.chan_conv(x)
        

    
class Descriminator(nn.Module):
    """
    GigaGAN SSR Descriminator
    :param in_channels: channels in the input Tensor.
    """

    def __init__(
        self,
        image_size=256,
        in_channels=3,
        out_channels=3,
        context_dim=1024,
        head_dim=64,
        max_channels=512,
        base_channels=8192,
        conv_blocks=1,
        pred_layers= 4,
        use_tree_pred=True,
        attention_resolutions=[8, 16, 32, 64],
        disable_spatial=[False, False, False, True],
        min_img = 8,
        attn_res_gain=0.5,
        embedding_channels=512,
        use_checkpoint=True,
        cdist_attention=True,
        use_sigmoid=True,
        conv_bias=True,
        msio=True,
        flatten_outputs=True,
        use_bn=True,
    ):
        super().__init__()
    
        # calculate the number of levels
        self.num_levels = int(math.log2(image_size))-2
        print(f"Building descriminator with {self.num_levels} levels {use_bn=}")
        
        self.projin = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.predictors = nn.ModuleList([])
        self.predskip = nn.ModuleList([])
        
        last_ch = None
        img_channels = 3

        self.msio = msio
        self.flatten_outputs = flatten_outputs

        
        # size tree for predictor
        if pred_layers == 4:
            sx = {512:30, 256:14, 128:6, 64:8, 32:8, 16:8, 8:4, 4:4, 2:2, 1:1}
        elif pred_layers == 3:
            sx = {512:64, 256:30, 128:14, 64:6, 32:8, 16:8, 8:4, 4:4, 2:2, 1:1}
        
        for level in range(self.num_levels):
            if image_size < min_img:
                img_channels = None
                
                
            image_size = image_size//2
            ch = min(base_channels // image_size, max_channels)
            print(f"level: {level}, img_size: {image_size}, ch: {ch}")
            
            # level0: -> [conv3N,BN,RU] -> [convNN,BN,RU]
            # level1: -------------------> [conv3N,BN,RU]
            # this suggests that each level should have a RGB input and a channel input
            # each will be trained independently
            

            force_bn = (level != 0) and (conv_blocks == 0)
            layers = [
                MultiConv(
                          img_channels=img_channels,
                          in_channels=last_ch, 
                          out_channels=ch,
                          kernel_size=4,
                          stride=2,
                          padding=2,
                          bias=False,
                         ),
                BatchNorm2d(ch, affine=True) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.1, inplace=False),
            ]
            
            for _ in range(conv_blocks):
                layers.append(
                    Conv2d(in_channels=ch, 
                          out_channels=ch,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=conv_bias,
                         )
                )
                layers.append(BatchNorm2d(ch, affine=True) if use_bn else nn.Identity())
                layers.append(nn.LeakyReLU(0.1, inplace=False))
            
            if image_size in attention_resolutions:
                layers.append(
                    SpatialTransformer(in_channels=ch, 
                        n_heads = ch // head_dim if head_dim != -1 else 1, 
                        d_head = head_dim if head_dim != -1 else ch,
                        depth=2,
                        dropout=0., 
                        context_dim=None,
                        disable_self_attn=True, 
                        use_linear=True,
                        use_checkpoint=use_checkpoint, 
                        dim=2, 
                        res_gain=attn_res_gain,
                        cdist_attention=cdist_attention, # should use L2 distance for attention instead of matmul)
                    )
                )
                       
            self.blocks.append(nn.Sequential(*layers)) 
            
            if (msio or level==0):
                print(" > Adding predictor")
                self.predictors.append(SimplePredictor(
                        input_nc=ch, n_layers=pred_layers,
                         embedding_channels=embedding_channels,
                         use_bn=use_bn,
                        ))
                
                self.predskip.append(Conv2d(ch, 1, 1, padding=0))
            
            last_ch = ch
            
        ch = min(base_channels // image_size, max_channels)
        # add the final layer
        
        self.blocks.append(nn.Sequential(
            # patch embed as 2x2
            Conv2d(last_ch, 
                      ch, 
                      kernel_size=2,
                      stride=2,
                      padding=0,
                      bias=True,
                     ),
                nn.LeakyReLU(0.1, inplace=False),
        ))
        
        print(" > Adding predictor")
        pch = 4*ch
        image_size = 1
        # always do a tree-predictor at the end
        self.predictors.append(SimplePredictor(
                    input_nc=pch, n_layers=pred_layers,
                    embedding_channels=embedding_channels,
                    use_bn=use_bn,
                ))
            
        # let's not add a norm step here
        self.predskip.append(nn.Sequential(
                Conv2d(pch, 1, 1, padding=0),
                #nn.AdaptiveAvgPool2d(output_size=(1,1)),
                #Conv2d(pch, 1, 1, padding=0),
        ))
        
        

    def forward(self, x):
        
        # get the input image size
        b,c,h,w = x.shape

        td = torch.tensor([])
        
        # obtain the start index
        index = self.num_levels - (int(math.log2(h)) - 2)
        
        # now, we want to create a list of logits
        logits = []
        for i in range(0, index):
            logits.append(None)
            
        for i in range(index, self.num_levels):
            x = self.blocks[i](x)
            if (self.msio) or (i == index):
                s = self.predskip[i](x)
                p = self.predictors[i](x, td)
                logits.append(p + s)
    
        # last layer
        x = self.blocks[-1](x)
        x = rearrange(x, 'b c h w -> b (c h w) 1 1')
        s = self.predskip[-1](x)
        p = self.predictors[-1](x, td)
        logits.append(p + s)

        # flatten the logits
        if self.flatten_outputs:
            return torch.cat([l.flatten() for l in logits])
        else:
            return logits
