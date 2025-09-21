import torch
import torch.nn.functional as F
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
import torch
import torch.nn as nn
import numpy as np
# from einops import rearrange
from util import instantiate_from_config
import torch
import numpy as np
from encoder_decoder import Encoder, Decoder
from helper_models import DiagonalGaussianDistribution

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 args=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=1e-4,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        super().__init__()
        self.args = args
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize encoder and decoder (you'll need to define these classes)
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # Initialize loss function (you'll need to implement instantiate_from_config)
        self.loss = instantiate_from_config(lossconfig)
        
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
        if monitor is not None:
            self.monitor = monitor
            
        # Initialize optimizers
        self._setup_optimizers()
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _setup_optimizers(self):
        """Setup optimizers for autoencoder and discriminator"""
        self.opt_ae = Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.learning_rate, 
            betas=(0.5, 0.9)
        )
        
        # Discriminator optimizer will be set up after loss is initialized
        if hasattr(self.loss, 'discriminator'):
            self.opt_disc = Adam(
                self.loss.discriminator.parameters(),
                lr=self.learning_rate, 
                betas=(0.5, 0.9)
            )

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Initialize model from checkpoint"""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, num_dwt_tensor):
        """Encode input to latent space"""
        h = self.encoder(x, num_dwt_tensor)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, num_dwt_tensor):
        """Decode latent representation to image"""
        z = self.post_quant_conv(z)
        dec = self.decoder(z, num_dwt_tensor)
        return dec

    def forward(self, input, num_dwt_tensor, sample_posterior=True):
        """Forward pass through autoencoder"""
        posterior = self.encode(input, num_dwt_tensor)
        # print("Posterior shape:", posterior.shape)
        if sample_posterior:
            z = posterior.sample()
            # print("Sampled z shape:", z.shape)
        else:
            z = posterior.mode()
        dec = self.decode(z, num_dwt_tensor)
        # print("Decoded shape:", dec.shape)
        return dec, posterior

    def get_input(self, batch, k, num_dwt_level):
        """Process input batch"""
        x = batch[k] if isinstance(batch, dict) else batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(memory_format=torch.contiguous_format).float()
        assert num_dwt_level.shape[0] == x.shape[0], "num_dwt_level batch size must match input batch size"
        num_dwt_tensor = torch.tensor(num_dwt_level, dtype=torch.float32).view(-1, 1).to(self.device)
        return x, num_dwt_tensor

    def training_step_ae(self, batch, num_dwt_level):
        if isinstance(batch, dict):
            inputs, num_dwt_tensor = self.get_input(batch, self.image_key, num_dwt_level)
        else:
            inputs, num_dwt_tensor = self.get_input(batch, None, num_dwt_level)
        
        inputs = inputs.to(self.device)
        num_dwt_tensor = num_dwt_tensor.to(self.device)
        reconstructions, posterior = self(inputs, num_dwt_tensor)  
        
        # Train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0,
            last_layer=self.get_last_layer(), split="train"
        )
        
        self.opt_ae.zero_grad()
        aeloss.backward()
        self.opt_ae.step()
        
        return aeloss, log_dict_ae

    def training_step_disc(self, batch, num_dwt_level):
        """Training step for discriminator"""
        if isinstance(batch, dict):
            inputs, num_dwt_tensor = self.get_input(batch, self.image_key, num_dwt_level)
        else:
            inputs, num_dwt_tensor = self.get_input(batch, None, num_dwt_level)
        
        inputs = inputs.to(self.device)
        num_dwt_tensor = num_dwt_tensor.to(self.device)
        reconstructions, posterior = self(inputs, num_dwt_tensor)
        
        # Train the discriminator
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1,
            last_layer=self.get_last_layer(), split="train"
        )
        
        self.opt_disc.zero_grad()
        discloss.backward()
        self.opt_disc.step()
        
        return discloss, log_dict_disc

    def validation_step(self, batch, num_dwt_level):
        """Validation step"""
        if isinstance(batch, dict):
            inputs, num_dwt_tensor = self.get_input(batch, self.image_key, num_dwt_level)
        else:
            inputs, num_dwt_tensor = self.get_input(batch, None, num_dwt_level)
        
        inputs = inputs.to(self.device)
        num_dwt_tensor = num_dwt_tensor.to(self.device)
        
        with torch.no_grad():
            reconstructions, posterior = self(inputs, num_dwt_tensor)
            
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, 0,
                last_layer=self.get_last_layer(), split="val"
            )
            
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, 1,
                last_layer=self.get_last_layer(), split="val"
            )
            
        return {
            "val_ae_loss": aeloss,
            "val_disc_loss": discloss,
            **log_dict_ae,
            **log_dict_disc
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch - MODIFIED to use args.log_freq"""
        self.train()
        total_ae_loss = 0
        total_disc_loss = 0
        
        log_freq = self.args.log_freq if self.args else 100
        
        for batch_idx, (batch, num_dwt_level) in enumerate(dataloader):
            # Train autoencoder
            ae_loss, log_dict_ae = self.training_step_ae(batch, num_dwt_level)
            total_ae_loss += ae_loss.item()
            
            # Train discriminator
            disc_loss, log_dict_disc = self.training_step_disc(batch, num_dwt_level)
            total_disc_loss += disc_loss.item()
            
            if batch_idx % log_freq == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: AE Loss: {ae_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}')
        
        avg_ae_loss = total_ae_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)
        
        return avg_ae_loss, avg_disc_loss

    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.eval()
        total_losses = {}
        
        for batch, num_dwt_level in dataloader:
            val_losses = self.validation_step(batch, num_dwt_level)
            for key, value in val_losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value.item() if torch.is_tensor(value) else value
        
        # Average the losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
            
        return total_losses

    def get_last_layer(self):
        """Get the last layer for gradient penalties"""
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, num_dwt_level, only_inputs=False, **kwargs):
        """Generate images for logging"""
        self.eval()
        log = dict()
        
        if isinstance(batch, dict):
            x, num_dwt_tensor = self.get_input(batch, self.image_key, num_dwt_level)
        else:
            x, num_dwt_tensor = self.get_input(batch, None, num_dwt_level)
        
        x = x.to(self.device)
        num_dwt_tensor = num_dwt_tensor.to(self.device)
        
        if not only_inputs:
            xrec, posterior = self(x, num_dwt_tensor)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()), num_dwt_tensor)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        """Convert to RGB for visualization"""
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def save_checkpoint(self, path, epoch, additional_info=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),  # This is correct for non-DDP
            'opt_ae_state_dict': self.opt_ae.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.opt_ae.load_state_dict(checkpoint['opt_ae_state_dict'])
        self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}, epoch {epoch}")
        return epoch

    def count_parameters(self, trainable_only=True):
        """
        Count the number of parameters in the model
        
        Args:
            trainable_only (bool): If True, count only trainable parameters
        
        Returns:
            dict: Dictionary with parameter counts for each component
        """
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                return sum(p.numel() for p in module.parameters())
        
        # Count parameters for each component
        encoder_params = count_params(self.encoder)
        decoder_params = count_params(self.decoder)
        quant_conv_params = count_params(self.quant_conv)
        post_quant_conv_params = count_params(self.post_quant_conv)
        
        # Count discriminator parameters if available
        discriminator_params = 0
        if hasattr(self.loss, 'discriminator'):
            discriminator_params = count_params(self.loss.discriminator)
        
        # Count other loss parameters (like logvar)
        loss_other_params = 0
        if hasattr(self.loss, 'logvar'):
            loss_other_params += self.loss.logvar.numel() if self.loss.logvar.requires_grad or not trainable_only else 0
        
        # Calculate totals
        autoencoder_params = encoder_params + decoder_params + quant_conv_params + post_quant_conv_params
        total_params = autoencoder_params + discriminator_params + loss_other_params
        
        param_dict = {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'quant_conv': quant_conv_params,
            'post_quant_conv': post_quant_conv_params,
            'autoencoder_total': autoencoder_params,
            'discriminator': discriminator_params,
            'loss_other': loss_other_params,
            'total': total_params
        }
        
        return param_dict

    def print_parameter_summary(self, trainable_only=True):
        """
        Print a formatted summary of model parameters
        
        Args:
            trainable_only (bool): If True, count only trainable parameters
        """
        params = self.count_parameters(trainable_only=trainable_only)
        
        param_type = "Trainable" if trainable_only else "Total"
        print(f"\n{'='*50}")
        print(f"{param_type} Parameters Summary")
        print(f"{'='*50}")
        
        # Format numbers with commas for readability
        def format_num(num):
            return f"{num:,}"
        
        print(f"Encoder:           {format_num(params['encoder']):>15}")
        print(f"Decoder:           {format_num(params['decoder']):>15}")
        print(f"Quant Conv:        {format_num(params['quant_conv']):>15}")
        print(f"Post Quant Conv:   {format_num(params['post_quant_conv']):>15}")
        print(f"{'-'*50}")
        print(f"Autoencoder Total: {format_num(params['autoencoder_total']):>15}")
        
        if params['discriminator'] > 0:
            print(f"Discriminator:     {format_num(params['discriminator']):>15}")
        
        if params['loss_other'] > 0:
            print(f"Loss (other):      {format_num(params['loss_other']):>15}")
        
        print(f"{'-'*50}")
        print(f"TOTAL:             {format_num(params['total']):>15}")
        print(f"{'='*50}")
        
        # Also print in millions/billions for easier reading
        total_m = params['total'] / 1e6
        if total_m >= 1000:
            print(f"Total: {total_m/1000:.2f}B parameters")
        else:
            print(f"Total: {total_m:.2f}M parameters")
        print()

    def get_model_size_mb(self):
        """
        Calculate the approximate model size in MB
        
        Returns:
            float: Model size in megabytes
        """
        total_params = self.count_parameters(trainable_only=False)['total']
        # Assuming float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb