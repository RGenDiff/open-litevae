import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


##########################################
##         GAN LOSS REGISTRY
##########################################
GAN_REGISTRY = {}

def register_gan_loss(name):
    """
    Decorator to register GAN loss functions.
    """
    def decorator(func):
        if name not in GAN_REGISTRY:
            GAN_REGISTRY[name] = {"g_func": None, "d_func": None}
        if "g_" in func.__name__:
            GAN_REGISTRY[name]["g_func"] = func
        elif "d_" in func.__name__:
            GAN_REGISTRY[name]["d_func"] = func
        else:
            raise ValueError("Function name must include 'g_' or 'd_' to specify type.")
        return func
    return decorator

def get_gan_loss(name):
    """
    Fetch generator and discriminator functions from the registry by name.
    """
    loss_pair = GAN_REGISTRY.get(name)
    if loss_pair is None or loss_pair["g_func"] is None or loss_pair["d_func"] is None:
        raise ValueError(f"Loss functions for '{name}' are not properly registered.")
    return loss_pair["g_func"], loss_pair["d_func"]

##########################################
##         HINGE : SNGAN
##########################################
# real is inf, fake is -inf

@register_gan_loss("hinge")
def g_hinge_loss(logits_fake):
    loss = -torch.mean(logits_fake)
    return loss

@register_gan_loss("hinge")
def d_hinge_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
    

##########################################
##         LOGISTIC 
##########################################

@register_gan_loss("logistic")
def g_logistic_loss(logits_fake):
    loss = torch.mean(F.softplus(-logits_fake))
    return loss

@register_gan_loss("logistic")
def d_logistic_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

##########################################
##         NSAT 
##########################################

@register_gan_loss("nsat")
def g_nsat_loss(logits_fake, eps=1e-6):
    loss = torch.log(logits_fake.sigmoid() + eps)
    return -torch.mean(loss)

@register_gan_loss("nsat")
def d_nsat_loss(logits_real, logits_fake, eps=1e-6):
    loss_real = torch.log(logits_real.sigmoid() + eps)
    loss_fake = torch.log(1.0 - logits_fake.sigmoid() + eps)
    loss = -0.5*(torch.mean(loss_real) + torch.mean(loss_fake) )
    return loss

##########################################
##         MSE : LSGAN
##########################################
# real is 1, fake is 0

@register_gan_loss("mse")
def g_mse_loss(logits_fake, eps=1e-6):
    loss = F.mse_loss(torch.ones_like(logits_fake), logits_fake)
    return loss

@register_gan_loss("mse")
def d_mse_loss(logits_real, logits_fake, eps=1e-6):
    loss_real = F.mse_loss(torch.ones_like(logits_fake), logits_fake)
    loss_fake = F.mse_loss(torch.zeros_like(logits_fake), logits_fake)
    loss = 0.5*(loss_real + loss_fake )
    return loss

##########################################
##         WGAN 
##########################################
# real is inf, fake is -inf

@register_gan_loss("wgan")
def g_wgan_loss(logits_fake, eps=1e-6):
    loss = -torch.mean(logits_fake)
    return loss

@register_gan_loss("wgan")
def d_wgan_loss(logits_real, logits_fake, eps=1e-6):
    loss_real = -torch.mean(loss_real)
    loss_fake = torch.mean(logits_fake)
    loss = 0.5*(loss_real + loss_fake )
    return loss

##########################################
##         BCE 
##########################################
# real is 1, fake is 0

@register_gan_loss("bce")
def g_bce_loss(logits_fake, eps=1e-6):
    loss = F.binary_cross_entropy(torch.ones_like(logits_fake), logits_fake.sigmoid())
    return loss

@register_gan_loss("bce")
def d_bce_loss(logits_real, logits_fake, eps=1e-6):
    loss_real = F.binary_cross_entropy(torch.ones_like(logits_fake), logits_fake.sigmoid())
    loss_fake = F.binary_cross_entropy(torch.zeros_like(logits_fake), logits_fake.sigmoid())
    loss = 0.5*(loss_real + loss_fake )
    return loss

