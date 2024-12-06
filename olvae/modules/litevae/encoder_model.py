import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch_dwt.functional import dwt2
from .unet_model import UNet

class LiteVAE_Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 dct_levels=3,
                 latent_dim=12,
                 image_size=256,
                 extractor_channels=32,
                 extractor_mult=[1, 2, 4],
                 extractor_resblocks=2,
                 aggregate_channels=32,
                 aggregate_mult=[1, 2, 4],
                 aggregate_resblocks=2,
                ):
        super(LiteVAE_Encoder, self).__init__()

        self.dct_levels = dct_levels
        
        # create the feature extractors
        self.extractors = nn.ModuleList([
            UNet(
                 in_channels = 4*in_channels,
                 channels = extractor_channels,
                 num_resblocks=extractor_resblocks,
                 channel_mult=extractor_mult,
                 image_size=image_size//2**l
                )
            for l in range(dct_levels)])

        # create the downsample layers - use ave pooling
        kernel_sizes = [2**l for l in range(dct_levels)][::-1]
        self.poolers = nn.ModuleList([
            nn.AvgPool2d(kernel_sizes[l]) 
            for l in range(dct_levels)])

        # compute the aggregate channel count
        self.aggregator = UNet(
                 in_channels = in_channels*4*dct_levels,
                 out_channels = latent_dim, # mu and logvar
                 channels = aggregate_channels,
                 num_resblocks=aggregate_resblocks,
                 channel_mult=aggregate_mult,
                 image_size=image_size//2**dct_levels
                )

    def _extract_DWT(self, x):

        # allocate an array for the outputs
        dwt_layers = []

        # set the first input to the x image
        cLL = x
        for _ in range(self.dct_levels):
            # compute the layer
            cLL, cLH, cHL, cHH = dwt2(cLL, "haar").unbind(dim=1)
            dwt_layers.append(torch.cat([cLL,cLH,cHL,cHH], dim=1))

        # return the list
        return dwt_layers
        
        
        
    def forward(self, x):
        
        # first we need to extract the DWT tensors
        with torch.no_grad(), autocast(enabled=False):
            dwt_layers = self._extract_DWT(x)

        # now extract the features
        features = []
        for l in range(self.dct_levels):
            h = self.extractors[l](dwt_layers[l].detach())
            features.append(self.poolers[l](h))

        # cat all of the features together in the channel dim
        h = torch.cat(features, dim=1)

        # no to the aggregator
        h = self.aggregator(h)

        return h