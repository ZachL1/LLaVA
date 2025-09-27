"""
CAVE model with integrated SD3.5 VAE for memory efficiency.

This implementation uses the SD3.5 VAE to process images in latent space instead of pixel space,
significantly reducing GPU memory usage.
"""

import torch
import torch.nn as nn
from cave.sd35.mmditx import MMDiTX, get_2d_sincos_pos_embed
from cave.sd35.sd3_impls import SDVAE
from cave.modules.base_model import BaseModel
from omegaconf import OmegaConf
from einops import rearrange
import json
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from safetensors import safe_open
import os


def load_into(ckpt, model, prefix, device, dtype=None):
    """Load weights from safetensors file to pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
            if obj is not None:
                try:
                    tensor = ckpt.get_tensor(key)
                    if dtype is not None:
                        tensor = tensor.to(dtype=dtype)
                    obj.data.copy_(tensor)
                except Exception as e:
                    print(f"Failed to load {key}: {e}")


class SD35VAEWrapper(nn.Module):
    """Wrapper for SD3.5 VAE with proper memory management."""
    
    def __init__(self, vae_path=None, dtype=torch.float16, device=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        
        # Create VAE model
        self.vae = SDVAE(dtype=dtype, device=device)
        
        # Load VAE weights if path provided
        if vae_path and os.path.exists(vae_path):
            self.load_vae_weights(vae_path)
        
        # Keep VAE on CPU by default to save memory
        self.vae = self.vae.cpu()
        self.vae.eval()
        
        # VAE scaling factor (SD3.5 uses 1.5305 typically)
        self.scale_factor = 1.5305
        
    def load_vae_weights(self, vae_path):
        """Load VAE weights from safetensors file."""
        with safe_open(vae_path, framework="pt", device="cpu") as f:
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.vae, prefix, "cpu", self.dtype)
    
    def encode(self, x):
        """Encode images to latent space."""
        # Store input dtype for consistency
        input_dtype = x.dtype
        
        # Move VAE to same device as input temporarily
        original_device = next(self.vae.parameters()).device
        self.vae = self.vae.to(x.device)
        
        # Normalize input from [-1, 1] to [0, 1] if needed
        if x.min() < -0.5:  # Assume input is in [-1, 1]
            x = (x + 1.0) / 2.0
        
        # Encode to latent - use autocast only if input is not already fp16
        if x.dtype == torch.float16:
            with torch.no_grad():
                latent = self.vae.encode(x)
                latent = latent * self.scale_factor
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.device.type == 'cuda'):
                with torch.no_grad():
                    latent = self.vae.encode(x)
                    latent = latent * self.scale_factor
        
        # Move VAE back to CPU
        self.vae = self.vae.to(original_device)
        
        # Ensure output dtype matches input for consistency
        if latent.dtype != input_dtype:
            latent = latent.to(dtype=input_dtype)
        
        return latent
    
    def decode(self, latent):
        """Decode latents to images."""
        # Store input dtype for consistency
        input_dtype = latent.dtype
        
        # Move VAE to same device as input temporarily
        original_device = next(self.vae.parameters()).device
        self.vae = self.vae.to(latent.device)
        
        # Scale latent
        latent_scaled = latent / self.scale_factor
        
        # Decode to image - use autocast only if input is not already fp16
        if latent_scaled.dtype == torch.float16:
            with torch.no_grad():
                x = self.vae.decode(latent_scaled)
                # Convert from [0, 1] to [-1, 1] for consistency with training
                x = x * 2.0 - 1.0
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=latent.device.type == 'cuda'):
                with torch.no_grad():
                    x = self.vae.decode(latent_scaled)
                    # Convert from [0, 1] to [-1, 1] for consistency with training
                    x = x * 2.0 - 1.0
        
        # Move VAE back to CPU
        self.vae = self.vae.to(original_device)
        
        # Ensure output dtype is appropriate for the model
        if x.dtype != input_dtype:
            x = x.to(dtype=input_dtype)
        
        return x


class CAVELatentEncoder(nn.Module):
    """
    MMDiTX-based encoder for CAVE working in latent space.
    
    Inputs:
        x: latent tensor from VAE (B, C_latent, H_latent, W_latent)
        c: condition (e.g., class label, pooled text embedding) (B, D)
        context: context tokens, number of context tokens can change
    
    Outputs:
        context: output context tokens for reconstruction
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Modify config for latent space (16 channels instead of 3)
        encoder_config = dict(config.model.encoder)
        encoder_config['in_channels'] = 16  # SD3.5 VAE latent channels
        encoder_config['patch_size'] = 2   # Smaller patch size for latent space
        
        # Store context token configuration before filtering
        self.max_context_tokens = encoder_config.pop('max_context_tokens', 256)
        self.min_context_tokens = encoder_config.pop('min_context_tokens', 1)
        self.default_context_tokens = encoder_config.pop('default_context_tokens', 16)
        
        # Convert string types
        if 'dtype' in encoder_config and isinstance(encoder_config['dtype'], str):
            if encoder_config['dtype'] == 'float16':
                encoder_config['dtype'] = torch.float16
            elif encoder_config['dtype'] == 'float32':
                encoder_config['dtype'] = torch.float32
            elif encoder_config['dtype'] == 'bfloat16':
                encoder_config['dtype'] = torch.bfloat16
            else:
                encoder_config.pop('dtype', None)
        
        if 'device' in encoder_config and encoder_config['device'] not in ['cuda', 'cpu']:
            encoder_config.pop('device', None)
        
        # Create MMDiTX model
        self.mmdit = MMDiTX(**encoder_config)
        
        # Initialize positional embeddings
        if hasattr(self.mmdit, 'pos_embed') and self.mmdit.pos_embed is not None:
            self._initialize_pos_embed()
        
        # MMDiTX uses hidden_size = 64 * depth
        hidden_size = 64 * config.model.encoder.get("depth", 12)
        
        # Create learnable context tokens for maximum possible length
        scale = hidden_size ** -0.5
        self.learnable_context_tokens = nn.Parameter(
            scale * torch.randn(self.max_context_tokens, hidden_size))
        
        # For VAE: project to mean and log_var
        self.to_mean = nn.Linear(hidden_size, hidden_size)
        self.to_log_var = nn.Linear(hidden_size, hidden_size)
        
        # # Initialize log_var projection to output small values initially
        # nn.init.zeros_(self.to_log_var.weight)
        # nn.init.constant_(self.to_log_var.bias, -5.0)  # Start with small variance
        
        self.context_proj = nn.Identity()
    
    def _initialize_pos_embed(self):
        """Initialize positional embeddings for encoder."""
        if hasattr(self.mmdit, 'pos_embed_max_size') and self.mmdit.pos_embed_max_size is not None:
            
            embed_dim = self.mmdit.pos_embed.shape[-1]
            grid_size = self.mmdit.pos_embed_max_size
            
            print(f"Encoder: Initializing pos_embed with grid_size={grid_size}, embed_dim={embed_dim}")
            print(f"Encoder: Current pos_embed shape: {self.mmdit.pos_embed.shape}")
            
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_size=grid_size,
                cls_token=False,
                extra_tokens=0,
                scaling_factor=getattr(self.mmdit, 'pos_embed_scaling_factor', None),
                offset=getattr(self.mmdit, 'pos_embed_offset', None)
            )
            
            pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
            print(f"Encoder: Generated pos_embed shape: {pos_embed_tensor.shape}")
            
            # Check if sizes match before copying
            if pos_embed_tensor.shape == self.mmdit.pos_embed.shape:
                self.mmdit.pos_embed.copy_(pos_embed_tensor)
                print("Encoder: Successfully copied pos_embed")
            else:
                print(f"Warning: Encoder position embedding shape mismatch. Expected {self.mmdit.pos_embed.shape}, got {pos_embed_tensor.shape}. Using default initialization.")
    
    def forward(self, x_latent, c=None, context=None, num_context_tokens=None, return_vae_params=False):
        """
        Forward pass for latent encoder.
        
        Args:
            x_latent: input latent tensor (B, 16, H_latent, W_latent)
            c: optional condition tensor (B, D)  
            context: optional context tokens (B, L, D)
            num_context_tokens: Number of context tokens to use (1-256). If None, uses default.
            return_vae_params: if True, return mean and log_var for VAE training
        
        Returns:
            If return_vae_params=False: context tokens (B, L, D)
            If return_vae_params=True: (mean, log_var) both (B, L, D)
        """
        batch_size = x_latent.size(0)
        
        # Use learnable context tokens if none provided
        if context is None:
            # Determine number of context tokens to use
            if num_context_tokens is None:
                if self.training:
                    # During training, randomly sample context length
                    num_context_tokens = torch.randint(
                        self.min_context_tokens, 
                        self.max_context_tokens + 1, 
                        (1,)
                    ).item()
                else:
                    # During inference/eval, use default unless specified
                    num_context_tokens = self.default_context_tokens
            
            # Validate context token count
            num_context_tokens = max(self.min_context_tokens, 
                                   min(num_context_tokens, self.max_context_tokens))
            
            # Get the appropriate number of context tokens
            context_tokens = self.learnable_context_tokens[:num_context_tokens]  # [num_tokens, hidden_size]
            context = context_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_tokens, hidden_size]
        
        # Create dummy timestep
        t = torch.zeros(batch_size, dtype=torch.long, device=x_latent.device)
        
        # Process latent through patch embedding
        x_embedded = self.mmdit.x_embedder(x_latent) + self.mmdit.cropped_pos_embed(x_latent.shape[-2:])
        
        # Create condition embedding
        c_mod = self.mmdit.t_embedder(t, dtype=x_latent.dtype)
        
        # Process through MMDiTX
        context_out, _ = self.mmdit.forward_core_with_concat(
            x=x_embedded,
            c_mod=c_mod,
            context=context
        )
        
        context_out = self.context_proj(context_out)
        
        if return_vae_params:
            # For VAE training: return mean and log_var
            mean = self.to_mean(context_out)
            log_var = self.to_log_var(context_out)
            return mean, log_var
        else:
            # For regular training/inference: return context tokens
            return context_out


class CAVELatentDecoder(nn.Module):
    """
    MMDiTX-based decoder for CAVE working in latent space.
    
    Inputs:
        x_latent: input latent tensor (B, 16, H_latent, W_latent)
        c: condition (B, D)
        context: context tokens from encoder
    
    Outputs:
        x_latent: output latent tensor for VAE decoding
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Modify config for latent space
        decoder_config = dict(config.model.decoder)
        decoder_config['in_channels'] = 16  # SD3.5 VAE latent channels
        decoder_config['patch_size'] = 2   # Smaller patch size for latent space
        
        # Convert string types
        if 'dtype' in decoder_config and isinstance(decoder_config['dtype'], str):
            if decoder_config['dtype'] == 'float16':
                decoder_config['dtype'] = torch.float16
            elif decoder_config['dtype'] == 'float32':
                decoder_config['dtype'] = torch.float32
            elif decoder_config['dtype'] == 'bfloat16':
                decoder_config['dtype'] = torch.bfloat16
            else:
                decoder_config.pop('dtype', None)
        
        if 'device' in decoder_config and decoder_config['device'] not in ['cuda', 'cpu']:
            decoder_config.pop('device', None)
        
        # Create MMDiTX model
        self.mmdit = MMDiTX(**decoder_config)
        
        # Initialize positional embeddings
        if hasattr(self.mmdit, 'pos_embed') and self.mmdit.pos_embed is not None:
            self._initialize_pos_embed()
        
        # Context projection if needed
        # MMDiTX uses hidden_size = 64 * depth
        encoder_hidden_size = 64 * config.model.encoder.get("depth", 12)
        decoder_hidden_size = 64 * config.model.decoder.get("depth", 12)
        
        if encoder_hidden_size != decoder_hidden_size:
            self.context_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.context_proj = nn.Identity()

        self.learnable_latent_tokens = nn.Parameter(
            torch.randn(16, 32, 32)
        )
    
    def _initialize_pos_embed(self):
        """Initialize positional embeddings for decoder."""
        if hasattr(self.mmdit, 'pos_embed_max_size') and self.mmdit.pos_embed_max_size is not None:
            
            embed_dim = self.mmdit.pos_embed.shape[-1]
            grid_size = self.mmdit.pos_embed_max_size
            
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_size=grid_size,
                cls_token=False,
                extra_tokens=0,
                scaling_factor=getattr(self.mmdit, 'pos_embed_scaling_factor', None),
                offset=getattr(self.mmdit, 'pos_embed_offset', None)
            )
            
            pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
            
            # Check if sizes match before copying
            if pos_embed_tensor.shape[1] == self.mmdit.pos_embed.shape[1]:
                self.mmdit.pos_embed.copy_(pos_embed_tensor)
            else:
                print(f"Warning: Decoder position embedding size mismatch. Expected {self.mmdit.pos_embed.shape[1]}, got {pos_embed_tensor.shape[1]}. Skipping pos_embed initialization.")
    
    def forward(self, x_latent, c=None, context=None):
        """
        Forward pass for latent decoder.
        
        Args:
            x_latent: input latent tensor (B, 16, H_latent, W_latent)
            c: optional condition tensor (B, D)
            context: context tokens from encoder (B, L, D)
        
        Returns:
            x_latent_out: output latent tensor (B, 16, H_latent, W_latent)
        """
        batch_size = x_latent.size(0)
        
        # Project context if needed
        if context is not None:
            context = self.context_proj(context)
        
        # Create dummy timestep
        t = torch.zeros(batch_size, dtype=torch.long, device=x_latent.device)
        
        # Process latent through patch embedding
        x_embedded = self.mmdit.x_embedder(x_latent) + self.mmdit.cropped_pos_embed(x_latent.shape[-2:])
        
        # Create condition embedding
        c_mod = self.mmdit.t_embedder(t, dtype=x_latent.dtype)
        
        # Process through MMDiTX
        _, x_out = self.mmdit.forward_core_with_concat(
            x=x_embedded,
            c_mod=c_mod,
            context=context
        )
        
        # Convert back to latent space using unpatchify
        x_out = self.mmdit.unpatchify(x_out, hw=x_latent.shape[-2:])
        
        return x_out


class CAVEWithVAE(BaseModel, PyTorchModelHubMixin):
    """
    CAVE model with integrated SD3.5 VAE for memory efficiency.
    
    This model works in latent space using the SD3.5 VAE, significantly reducing
    GPU memory usage compared to working directly with high-resolution images.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract VAE path from config
        vae_path = getattr(config.model, 'vae_path', None)
        if hasattr(config.model, 'sd3_model_path') and config.model.sd3_model_path:
            # Try to use the main SD3.5 model file for VAE weights
            vae_path = config.model.sd3_model_path
        
        # Initialize VAE wrapper
        self.vae = SD35VAEWrapper(
            vae_path=vae_path,
            dtype=torch.float16,  # Use fp16 for memory efficiency
            device=None  # Will be moved to GPU as needed
        )
        
        # Freeze VAE parameters to save memory and avoid gradient issues
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Initialize encoder and decoder for latent space
        self.encoder = CAVELatentEncoder(config)
        self.decoder = CAVELatentDecoder(config)
        
        # Diffusion decoder (optional, for diffusion training)
        self.use_diffusion = getattr(config.model, 'use_diffusion', False)
        self.diffusion_decoder = None
        
        if self.use_diffusion and hasattr(config.model, 'sd3_model_path') and config.model.sd3_model_path:
            # Initialize diffusion decoder with SD3.5 model
            # This would require additional implementation for diffusion training
            print("Diffusion decoder initialization would go here")
    
    def encode_to_latent(self, x):
        """Encode images to latent space using VAE."""
        with torch.no_grad():  # Ensure VAE encoding doesn't require gradients
            latent = self.vae.encode(x)
        return latent.detach()  # Detach to prevent gradient flow through VAE
    
    def decode_from_latent(self, latent):
        """Decode latents to images using VAE."""
        # Note: We need gradients to flow through the latent input for training
        # VAE parameters are frozen, but gradients can flow through the latent
        return self.vae.decode(latent)
    
    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick for VAE.
        
        Args:
            mean: mean of the latent distribution (B, L, D)
            log_var: log variance of the latent distribution (B, L, D)
        
        Returns:
            sampled latent: sampled from the distribution (B, L, D)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, use_diffusion=False, train_diffusion=False, return_latents=False, num_context_tokens=None, use_vae_training=False, use_learnable=False):
        """
        Forward pass for CAVE with VAE.
        
        Args:
            x: input images (B, 3, H, W) in range [-1, 1]
            use_diffusion: whether to use diffusion decoder
            train_diffusion: whether to train with diffusion (alias for use_diffusion)
            return_latents: whether to return intermediate latents
            num_context_tokens: Number of context tokens to use (1-256). If None, uses default.
            use_vae_training: whether to use VAE training with KL loss (default: True)
        
        Returns:
            reconstructed: reconstructed images (B, 3, H, W)
            extra_info: dict with additional information
        """
        extra_info = {}
        
        # Handle train_diffusion parameter (alias for use_diffusion)
        if train_diffusion:
            use_diffusion = True
        
        # Encode to latent space (detached - no gradients for input)
        x_latent = self.encode_to_latent(x)
        extra_info['input_latent'] = x_latent
        
        # Ensure latent has same dtype as model parameters for mixed precision compatibility
        model_dtype = next(self.encoder.parameters()).dtype
        if x_latent.dtype != model_dtype:
            x_latent = x_latent.to(dtype=model_dtype)
        
        # Create a version with gradients enabled for training
        x_latent_trainable = x_latent.detach().clone().requires_grad_(True)
        
        # Encode latent to context tokens with variable length support
        if use_vae_training:
            # VAE training: get mean and log_var for reparameterization
            mean, log_var = self.encoder(x_latent_trainable, num_context_tokens=num_context_tokens, return_vae_params=True)
            context = self.reparameterize(mean, log_var)
            extra_info['mean'] = mean
            extra_info['log_var'] = log_var
        else:
            # Regular training/inference: get context tokens directly
            context = self.encoder(x_latent_trainable, num_context_tokens=num_context_tokens, return_vae_params=False)
        
        extra_info['context'] = context
        extra_info['num_context_tokens'] = context.shape[1] if context is not None else 0
        
        if use_diffusion and self.diffusion_decoder is not None:
            # Use diffusion decoder (to be implemented)
            raise NotImplementedError("Diffusion decoder not yet implemented")
        else:
            # Use reconstruction decoder
            # Create noise latent for decoder input (proper generative approach)
            noise_latent = torch.randn_like(x_latent_trainable, dtype=model_dtype)
            if use_learnable:
                noise_latent = self.decoder.learnable_latent_tokens.unsqueeze(0).expand(x_latent_trainable.shape[0], -1, -1, -1)
            
            # Decoder learns to reconstruct the original latent from noise + context
            reconstructed_latent = self.decoder(noise_latent, context=context)
            extra_info['reconstructed_latent'] = reconstructed_latent
            
            # For training loss, we'll compare latents directly
            # For visualization, decode to image space - ensure consistent dtype
            if reconstructed_latent.dtype != x_latent.dtype:
                decode_latent = reconstructed_latent.to(dtype=x_latent.dtype)
            else:
                decode_latent = reconstructed_latent
                
            reconstructed = self.decode_from_latent(decode_latent)
            
            # Add latent reconstruction loss info
            extra_info['latent_target'] = x_latent  # Target for latent loss
        
        if return_latents:
            extra_info['latents'] = {
                'input_latent': x_latent,
                'reconstructed_latent': reconstructed_latent
            }
        
        return reconstructed, extra_info
    
    def encode(self, x, num_context_tokens=None, use_vae_training=False):
        """Encode images to context tokens via latent space."""
        x_latent = self.encode_to_latent(x)
        if use_vae_training:
            mean, log_var = self.encoder(x_latent, num_context_tokens=num_context_tokens, return_vae_params=True)
            context = self.reparameterize(mean, log_var)
        else:
            context = self.encoder(x_latent, num_context_tokens=num_context_tokens, return_vae_params=False)
        return context
    
    def decode(self, context, latent_shape=None, noise=None, use_learnable=False):
        """Decode context tokens to images via latent space."""
        if noise is None and latent_shape is None:
            raise ValueError("Either noise or latent_shape must be provided")
        
        if noise is None:
            # Create noise with specified shape
            noise = torch.randn(latent_shape, device=context.device, dtype=context.dtype)
        if use_learnable:
            noise = self.decoder.learnable_latent_tokens.unsqueeze(0).expand(latent_shape[0], -1, -1, -1)
        
        # Decode context to latent
        reconstructed_latent = self.decoder(noise, context=context)
        
        # Decode latent to image
        reconstructed = self.decode_from_latent(reconstructed_latent)
        
        return reconstructed
    
    def num_parameters(self, only_trainable=False, exclude_embeddings=False):
        """Count model parameters, excluding VAE."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() 
                           if not only_trainable or p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters()
                           if not only_trainable or p.requires_grad)
        
        total = encoder_params + decoder_params
        
        if self.diffusion_decoder is not None:
            diffusion_params = sum(p.numel() for p in self.diffusion_decoder.parameters()
                                 if not only_trainable or p.requires_grad)
            total += diffusion_params
        
        return total
    
    def get_latent_shape(self, image_shape):
        """Get the shape of latents for given image shape."""
        B, C, H, W = image_shape
        # SD3.5 VAE has 8x downsampling factor
        return (B, 16, H // 8, W // 8)

    def load_checkpoint(self, checkpoint_dir, kl=False, use_learnable=False):
        index_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
        # Load the state_dict from all shards
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data['weight_map']
        all_shards = sorted(list(set(weight_map.values())))
        
        import tqdm, gc
        state_dict = {}
        for shard_file in tqdm.tqdm(all_shards, desc="Loading shards"):
            shard_path = os.path.join(checkpoint_dir, shard_file)
            shard_data = torch.load(shard_path, map_location='cpu', weights_only=True)
            state_dict.update(shard_data)
            # Free memory as we go
            del shard_data
            gc.collect()
        
        print(f"Loaded {len(state_dict)} total parameters from shards.")

        # Clean state dict keys to ensure they match the model structure.
        clean_state_dict = {}
        for key, value in state_dict.items():
            if not kl and (key.startswith('encoder.to_mean.') or key.startswith('encoder.to_log_var.')):
                print("[WARNING] using a checkpoint with kl training!!!")
                continue
            if not use_learnable and key.startswith("decoder.learnable_latent_tokens"):
                print("[WARNING] using a checkpoint with learnable token training!!!")
                continue
            new_key = key
            # Handle the common 'module.' prefix from models saved with DataParallel.
            if new_key.startswith('module.'):
                new_key = new_key.replace('module.', '', 1)
            clean_state_dict[new_key] = value

        # First, attempt to load the checkpoint with strict=True for perfect matching.
        print("Attempting to load state dict with strict checking...")
        self.load_state_dict(clean_state_dict, strict=False)