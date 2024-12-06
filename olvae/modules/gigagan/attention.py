# Based on the transformer block code form https://github.com/CompVis/stable-diffusion
import math

import numpy as np
import torch
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
    
_ATTN_PRECISION ="fp32"

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
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(32, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class normalization(GroupNorm32):
    pass

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., bias=True):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="tanh"),
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(inner_dim, dim_out, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class L2CrossAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0., context_dim=None, use_dot=False, qkv_bias=True):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads, with use_dot={use_dot}.")
        inner_dim = dim_head * heads
        no_k = context_dim
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_dot = use_dot

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)   # K and Q are tied in L2
        self.to_v = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        
    
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        v = self.to_v(x)
        b, _, _ = q.shape
        
        q, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, v))
        
        # tie k and q
        k = q
        
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q = q.float()
                AB = torch.matmul(q, k.transpose(-1, -2))
                AA = torch.sum(q ** 2, -1, keepdim=True)
                BB = AA.transpose(-1, -2)    # Since query and key are tied.
                sim = -(AA - 2 * AB + BB) * self.scale
        else:
            raise Exception("Only support FP32 attention with L2")
        
        del q
    

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "cdist": L2CrossAttention, #L2 distance attention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, use_checkpoint=True,
                 disable_self_attn=False, cdist_attention=False):
        super().__init__()
        
        
        if cdist_attention:
            sattn_mode = "cdist"
            cattn_mode = "cdist" 
        else:
            raise Exception("Only support cdist attention in disc!")

        assert sattn_mode in self.ATTENTION_MODES
        assert cattn_mode in self.ATTENTION_MODES
        sattn_cls = self.ATTENTION_MODES[sattn_mode]
        cattn_cls = self.ATTENTION_MODES[cattn_mode]
        
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            print("Transformer Block disabling self attention")
            self.attn1 = None
            self.sa_norm = None
        else:
            print("Transformer Block enabling self attention")
            self.sa_norm = LayerNorm(dim)
            self.attn1 = sattn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=None, use_dot=not cdist_attention ) 
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = cattn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.ff_norm = LayerNorm(dim) #don't need to duplicate the linear layer
        self.ca_norm = LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None, mask=None, height=None):
        if self.training and self.use_checkpoint:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states = torch.utils.checkpoint.checkpoint(
                    self._forward,
                    x,
                    context,
                    mask,
                    **ckpt_kwargs,
                )
            return hidden_states
        else:
            return self._forward(x, context, mask, height)
        

    def _forward(self, x, context=None, mask=None, height=None):
        if not self.disable_self_attn:
            x = self.attn1(self.sa_norm(x), context=None) + x
        x = self.attn2(self.ca_norm(x), context=context, mask=mask) + x
        x = self.ff(self.ff_norm(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dim=2, 
                 res_gain=1.0,
                 cdist_attention=False, # should use L2 distance for attention instead of matmul
                ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        else:
            context_dim = [None]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels) if dim != 1 else LayerNorm(in_channels)
        self.dim = dim
        self.res_gain = res_gain
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[0],
                                   disable_self_attn=disable_self_attn, use_checkpoint=use_checkpoint,
                                  cdist_attention=cdist_attention)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, mask=None):
        
        #print("[[xformer]]")
        
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
                
        x_in = x
        x = self.norm(x)
        
        if self.dim == 1:
            b, h, c = x.shape
        else:
            b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        x = self.proj_in(x)
        
        #print(x.shape)
        
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[0], mask=mask, height=h)
            

        x = self.proj_out(x)
        
        if self.dim != 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            
        return x + x_in*self.res_gain
