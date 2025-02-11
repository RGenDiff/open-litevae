# based on the implementation from https://github.com/CompVis/latent-diffusion [MIT license]

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import itertools
from tqdm import tqdm

from pytorch_lightning.utilities.distributed import rank_zero_only
from olvae.modules.distributions import DiagonalGaussianDistribution

from olvae.util import instantiate_from_config, count_params
from contextlib import contextmanager
from olvae.modules.ema import LitEma

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LiteAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 lossconfig,
                 embed_dim,
                 use_quant=True,
                 use_ema = False,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 save_top_k=3,
                 freeze_latent_space=False,
                 monitor=None,
                 mon_mode=None,
                 disc_lr_ratio=1.0,
                 use_dyn_loss=False,
                 aux_loss_weight=0.0,
                 aux_start=0, # negative means hard start, positive means ramp up
                 aux_end=-1, # if positive, then sets an end step
                 ):
        super().__init__()

        self.bad_state = False # used for trainer to stop corrupting NaNs into EMA
        # set config saves
        self.image_key = image_key
        self.use_ema = use_ema
        self.use_qaunt = use_quant
        self.freeze_latent_space = freeze_latent_space
        self.disc_lr_ratio = disc_lr_ratio
        self.use_dyn_loss = use_dyn_loss
        self.aux_start = aux_start
        self.aux_end = aux_end
        self.aux_loss_weight = aux_loss_weight

        
        # set the output channels for the encoder if use_qaunt is false
        if not use_quant:
            encoder_config.params.latent_dim = 2*embed_dim

        # instantiate the encoder, decoder, and loss
        self.encoder = instantiate_from_config(encoder_config)
        count_params(self.encoder, verbose=True)
        self.decoder = instantiate_from_config(decoder_config)
        count_params(self.decoder, verbose=True)
        self.loss = instantiate_from_config(lossconfig)

        # setup the quantization
        z_channels = 2*embed_dim
        self.quantizer = torch.nn.Module()
        self.quantizer.quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1) if use_quant else torch.nn.Identity()
        self.quantizer.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1) if use_quant else torch.nn.Identity()
        self.embed_dim = embed_dim

        # freeze the encoder and quantizer if we are freezing the latent space
        if freeze_latent_space:
            self.freeze_latents()

        # setup monitoring
        self.save_top_k = save_top_k
        if mon_mode is not None:
            self.mon_mode = mon_mode
        if monitor is not None:
            self.monitor = monitor

        # setup EMA
        if self.use_ema:
            self.encoder_ema = LitEma(self.encoder)
            self.decoder_ema = LitEma(self.decoder)
            self.quantizer_ema = LitEma(self.quantizer)
            # buffer count
            count = len(list(self.encoder_ema.buffers()))
            count += len(list(self.decoder_ema.buffers()))
            count += len(list(self.quantizer.buffers()))
            # print the feedback
            print(f"Keeping EMAs of {count}.")

        # load from checkpoint if applicable
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def freeze_latents(self):
        # quant conv which establishes mu and logvar
        self.quantizer.quant_conv.eval()
        self.quantizer.quant_conv.train = disabled_train
        for param in self.quantizer.quant_conv.parameters():
            param.requires_grad = False

        # encoder which feeds into the quant conv
        # or it establishes mu and logvar if quant is disabled
        self.encoder.eval()
        self.encoder.train = disabled_train
        for param in self.encoder.parameters():
            param.requires_grad = False

    
    @contextmanager
    def ema_scope(self, context=None, verbose=True):
        if self.use_ema:
            self.model_ema.store(self.model.parameters(), verbose=verbose)
            self.model_ema.copy_to(self.model)
            if (context is not None) and verbose:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters(), verbose=verbose)
                if (context is not None) and verbose:
                    print(f"{context}: Restored training weights")
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def load_state_dict(self, state_dict, strict=False):
        """Override to ignore keys that do not match."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    print('While copying the parameter named {}, whose dimensions in the model are '
                          '{} and whose dimensions in the checkpoint are {}, an exception occurred: {}.'
                          .format(name, own_state[name].size(), param.size(), e))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        return [], []

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quantizer.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.quantizer.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]

        # if for some reason we have a grayscale image in B,H,W, then convert to B,H,W,C
        if len(x.shape) == 3:
            x = x[..., None]

        # convert to B,C,H,W
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def hold_disc_grads(self, freeze):
        pass
        # @TODO: to be correct during batch accumulation, we should freeze 
        #        the discriminator gradients. However, empirical results show
        #        that the models tend to achieve better metrics when not frozen
        #self.loss.discriminator.requires_grad_(not freeze)
        #self.loss.discriminator.train(not freeze)

    def on_train_batch_end(self, *args, **kwargs):
        
        # Check for NaNs/Infs in gradients
        bad_state_detected = False
        for name, param in self.named_parameters():
            if (torch.isnan(param.data).any() or torch.isinf(param.data).any() or 
                param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())):
                print(f'Detected NaN or Inf in {name}, recovering from last checkpoint')

                bad_state_detected = True
                break
       
        # if there was a bade state, then error out
        if bad_state_detected:
            # let's just error out for now
            self.bad_state = True
            assert False, "Network entered a bad training state with inf or NaN values"

        # at this point, the bad_state_detected == False

        # if we're not using EMA, then there's nothing to do
        if not self.use_ema:
            return

        # update the EMAs
        if not self.freeze_latent_space:
            self.encoder_ema(self.encoder) 
        self.decoder_ema(self.decoder)
        self.quantizer_ema(self.quantizer)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False) 
        
        if optimizer_idx == 0:
            # freeze the disc due to gradient accumulation
            self.hold_disc_grads(True)

            # step get the reconstructions
            reconstructions, posterior = self(inputs)

            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # get the auxloss if enabled
            if self.aux_loss_weight > 0.0:
                if (self.aux_end > 0) and (self.global_step > self.aux_end):
                    aux_mult = 0.0
                elif self.aux_start < 0:
                    aux_mult = 0.0 if self.global_step < (-self.aux_start) else 1.0
                elif self.aux_start > 0:
                    aux_mult = np.clip(self.global_step/self.aux_start, 0.0, 1.0)
                else:
                    aux_mult = 1.0
            
                aux_loss = self.decoder.get_aux_loss()*self.aux_loss_weight*aux_mult
                log_dict_ae.update({'train/aux_loss': aux_loss.detach().mean()})

                aeloss = aeloss + aux_loss.mean()
            
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

            self.log('rec_loss', log_dict_ae['train/rec_loss'], 
                     prog_bar=True, logger=False, on_step=True, on_epoch=False)

            # unfreeze disc
            self.hold_disc_grads(False)

            return aeloss

        if optimizer_idx == 1:

            # we can generate with no grad
            with torch.no_grad():
                reconstructions, posterior = self(inputs)

            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        inputs = self.get_input(batch, self.image_key)

        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict



    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters())+
                                  list(self.quantizer.parameters())+
                                  list(self.loss.get_trainable_autoencoder_parameters()),
                                  lr=lr, betas=(0.5, 0.9))

    
        opt_disc = torch.optim.Adam(list(self.loss.get_trainable_parameters()),
                          lr=lr*self.disc_lr_ratio, betas=(0.5, 0.9))


        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        if self.use_dyn_loss:
            return self.decoder.conv_out.weight
        else:
            return None

    def sample_images(self, x):
        xrec, posterior = self(x)
                
        xsample = self.decode(torch.randn_like(posterior.sample()))
        
        return xrec, xsample

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        if not only_inputs:
            xrec, xsample = self.sample_images(x)
            log["samples"] = xsample
            log["reconstructions"] = xrec 
            # also save the differences, rescaled to -1 to 1
            log["rec_errors"] = torch.abs(x - xrec).clip(0,1)*2.0 - 1.0 

        log["inputs"] = x
        return log
    



