import torch
import torch.nn as nn
from cave.sd35.mmditx import MMDiTX, PatchEmbed, JointBlock, FinalLayer
from cave.modules.base_model import BaseModel
from omegaconf import OmegaConf
from einops import rearrange
import json
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin


class CAVEEncoder(nn.Module):
    '''
    MMDiTX-based encoder for CAVE.

    Inputs:
        x: input image
        c: condition (e.g., class label, pooled text embedding) (,D)
        context: context tokens, number of context tokens can change

    Outputs:
        context: output context tokens, same shape as input context
    '''
    def __init__(self, config):
        super().__init__()
        
        # Prepare encoder config with proper dtype and device conversion
        encoder_config = dict(config.model.encoder)
        
        # Convert string dtype to torch dtype
        if 'dtype' in encoder_config:
            if isinstance(encoder_config['dtype'], str):
                if encoder_config['dtype'] == 'float16':
                    encoder_config['dtype'] = torch.float16
                elif encoder_config['dtype'] == 'float32':
                    encoder_config['dtype'] = torch.float32
                elif encoder_config['dtype'] == 'bfloat16':
                    encoder_config['dtype'] = torch.bfloat16
                else:
                    encoder_config.pop('dtype', None)
            elif encoder_config['dtype'] is None:
                encoder_config.pop('dtype', None)
        
        # Convert string device to proper device or remove it
        if 'device' in encoder_config:
            if isinstance(encoder_config['device'], str):
                if encoder_config['device'] in ['cuda', 'cpu']:
                    pass  # Keep the string
                else:
                    encoder_config.pop('device', None)
            elif encoder_config['device'] is None:
                encoder_config.pop('device', None)
        
        # Create MMDiTX model with encoder configuration
        self.mmdit = MMDiTX(**encoder_config)
        
        # Initialize positional embeddings if pos_embed_max_size is provided
        if hasattr(self.mmdit, 'pos_embed') and self.mmdit.pos_embed is not None:
            self._initialize_pos_embed()
        
        # Initialize learnable context tokens
        self.num_context_tokens = 16
        hidden_size = config.model.encoder.get("depth", 12) * 64  # MMDiTX uses depth * 64 as hidden_size
        
        # Scale initialization similar to TiTok
        scale = hidden_size ** -0.5
        self.learnable_context_tokens = nn.Parameter(
            scale * torch.randn(self.num_context_tokens, hidden_size))
        
        # No context projection needed - keep context in the same dimension as MMDiTX
        self.context_proj = nn.Identity()

    def _initialize_pos_embed(self):
        """Initialize positional embeddings for the MMDiTX model."""
        if hasattr(self.mmdit, 'pos_embed_max_size') and self.mmdit.pos_embed_max_size is not None:
            from cave.sd35.mmditx import get_2d_sincos_pos_embed
            import numpy as np
            
            # Calculate positional embeddings
            embed_dim = self.mmdit.pos_embed.shape[-1]  # hidden_size
            grid_size = self.mmdit.pos_embed_max_size
            
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_size=grid_size,
                cls_token=False,
                extra_tokens=0,
                scaling_factor=self.mmdit.pos_embed_scaling_factor,
                offset=self.mmdit.pos_embed_offset
            )
            
            # Convert to torch tensor and copy to the buffer
            pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.mmdit.pos_embed.copy_(pos_embed_tensor)

    def forward(self, x, c=None, context=None):
        """
        Forward pass for CAVE encoder.
        
        Args:
            x: input image tensor (B, C, H, W)
            c: optional condition tensor (B, D)
            context: optional context tokens (B, L, D). If None, uses learnable tokens.
        
        Returns:
            context: output context tokens (B, L, D)
        """
        batch_size = x.size(0)
        
        # Use learnable context tokens if none provided
        if context is None:
            context = self.learnable_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create dummy timestep (not used in encoder, but required by MMDiTX interface)
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Process image through patch embedding and add positional encoding
        x_embedded = self.mmdit.x_embedder(x) + self.mmdit.cropped_pos_embed(x.shape[-2:])
        
        # Create condition embedding
        c_mod = self.mmdit.t_embedder(t, dtype=x.dtype)
        # Don't try to use y_embedder since adm_in_channels is None in our config
        
        # Use forward_core_with_concat to get modulated context
        # This processes the image and modulates the context tokens
        context_out, _ = self.mmdit.forward_core_with_concat(
            x=x_embedded,
            c_mod=c_mod,
            context=context
        )
        
        # Project to desired context dimension
        context_out = self.context_proj(context_out)
        
        return context_out


class CAVEDecoder(nn.Module):
    '''
    MMDiTX-based decoder for CAVE.

    Inputs:
        x: noise latent of the same shape as the output image or the input image
           (e.g., for image editing, x is the input image; for reconstruction/generation, x is noise)
        c: condition (e.g., class label, pooled text embedding) (,D)
        context: context tokens from CAVEEncoder.

    Outputs:
        x: output image, same shape as input image

        Refer to the SD35 codebase for implementation details.
        Note CAVEDecoder is for resconstruction tasks, not for diffusion. Thus no timesteps.
    '''
    def __init__(self, config):
        super().__init__()
        
        # For reconstruction decoder, force 3 channels for RGB images
        # Create a modified config for reconstruction
        reconstruction_config = OmegaConf.create(dict(config))
        reconstruction_config.model.decoder.in_channels = 3  # RGB format
        
        # Prepare decoder config with proper dtype and device conversion
        decoder_config = dict(reconstruction_config.model.decoder)
        
        # Convert string dtype to torch dtype
        if 'dtype' in decoder_config:
            if isinstance(decoder_config['dtype'], str):
                if decoder_config['dtype'] == 'float16':
                    decoder_config['dtype'] = torch.float16
                elif decoder_config['dtype'] == 'float32':
                    decoder_config['dtype'] = torch.float32
                elif decoder_config['dtype'] == 'bfloat16':
                    decoder_config['dtype'] = torch.bfloat16
                else:
                    decoder_config.pop('dtype', None)
            elif decoder_config['dtype'] is None:
                decoder_config.pop('dtype', None)
        
        # Convert string device to proper device or remove it
        if 'device' in decoder_config:
            if isinstance(decoder_config['device'], str):
                if decoder_config['device'] in ['cuda', 'cpu']:
                    pass  # Keep the string
                else:
                    decoder_config.pop('device', None)
            elif decoder_config['device'] is None:
                decoder_config.pop('device', None)
        
        # Create MMDiTX model with decoder configuration
        self.mmdit = MMDiTX(**decoder_config)
        
        # Initialize positional embeddings if pos_embed_max_size is provided
        if hasattr(self.mmdit, 'pos_embed') and self.mmdit.pos_embed is not None:
            self._initialize_pos_embed()
        
        # Context projection from encoder dimension to decoder dimension if needed
        encoder_hidden_size = config.model.encoder.get("depth", 12) * 64
        decoder_hidden_size = config.model.decoder.get("depth", 12) * 64
        
        if encoder_hidden_size != decoder_hidden_size:
            self.context_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.context_proj = nn.Identity()

    def _initialize_pos_embed(self):
        """Initialize positional embeddings for the MMDiTX model."""
        if hasattr(self.mmdit, 'pos_embed_max_size') and self.mmdit.pos_embed_max_size is not None:
            from cave.sd35.mmditx import get_2d_sincos_pos_embed
            import numpy as np
            
            # Calculate positional embeddings
            embed_dim = self.mmdit.pos_embed.shape[-1]  # hidden_size
            grid_size = self.mmdit.pos_embed_max_size
            
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_size=grid_size,
                cls_token=False,
                extra_tokens=0,
                scaling_factor=self.mmdit.pos_embed_scaling_factor,
                offset=self.mmdit.pos_embed_offset
            )
            
            # Convert to torch tensor and copy to the buffer
            pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.mmdit.pos_embed.copy_(pos_embed_tensor)

    def forward(self, x, c=None, context=None):
        """
        Forward pass for CAVE decoder.
        
        Args:
            x: noise latent or input image tensor (B, C, H, W)
            c: optional condition tensor (B, D)
            context: context tokens from encoder (B, L, D)
        
        Returns:
            output: reconstructed image (B, C, H, W)
        """
        batch_size = x.size(0)
        
        # Project context to decoder dimension
        if context is not None:
            context = self.context_proj(context)
        
        # Create dummy timestep (not used in reconstruction, but required by MMDiTX interface)
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Process input through decoder
        # Don't pass y=c since the MMDiTX model doesn't have y_embedder when adm_in_channels is None
        output = self.mmdit(x, t, context=context)
        
        return output

class CAVEDiffusionDecoder(nn.Module):
    '''
    MMDiTX-based diffusion decoder for CAVE.

    Inputs:
        x: noise latent of the same shape as the output image or the input image
           (e.g., for image editing, x is the input image; for reconstruction/generation, x is noise)
        c: condition (e.g., class label, pooled text embedding) (,D)
        context: context tokens from CAVEEncoder.

    Outputs:
        x: output image, same shape as input image

        Refer to the SD35 codebase for the diffusion process details.
        Note CAVEDiffusionDecoder is for diffusion tasks, not for reconstruction. Thus uses timesteps.
    '''
    def __init__(self, config):
        super().__init__()
        
        # For diffusion decoder, we need to use 16 channels like SD3.5
        # Create a modified config for diffusion
        diffusion_config = OmegaConf.create(dict(config))
        diffusion_config.model.decoder.in_channels = 16  # SD3.5 format
        
        # Prepare decoder config with proper dtype and device conversion
        decoder_config = dict(diffusion_config.model.decoder)
        
        # Convert string dtype to torch dtype
        if 'dtype' in decoder_config:
            if isinstance(decoder_config['dtype'], str):
                if decoder_config['dtype'] == 'float16':
                    decoder_config['dtype'] = torch.float16
                elif decoder_config['dtype'] == 'float32':
                    decoder_config['dtype'] = torch.float32
                elif decoder_config['dtype'] == 'bfloat16':
                    decoder_config['dtype'] = torch.bfloat16
                else:
                    decoder_config.pop('dtype', None)
            elif decoder_config['dtype'] is None:
                decoder_config.pop('dtype', None)
        
        # Convert string device to proper device or remove it
        if 'device' in decoder_config:
            if isinstance(decoder_config['device'], str):
                if decoder_config['device'] in ['cuda', 'cpu']:
                    pass  # Keep the string
                else:
                    decoder_config.pop('device', None)
            elif decoder_config['device'] is None:
                decoder_config.pop('device', None)
        
        # Create MMDiTX model with decoder configuration for diffusion
        self.mmdit = MMDiTX(**decoder_config)
        
        # Initialize positional embeddings if pos_embed_max_size is provided
        if hasattr(self.mmdit, 'pos_embed') and self.mmdit.pos_embed is not None:
            self._initialize_pos_embed()
        
        # Context projection from encoder dimension to decoder dimension if needed
        encoder_hidden_size = config.model.encoder.get("depth", 12) * 64
        decoder_hidden_size = config.model.decoder.get("depth", 12) * 64
        
        if encoder_hidden_size != decoder_hidden_size:
            self.context_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.context_proj = nn.Identity()
        
        # Diffusion model sampling setup
        from cave.sd35.sd3_impls import ModelSamplingDiscreteFlow
        self.model_sampling = ModelSamplingDiscreteFlow(shift=1.0)
        
    def apply_model(self, x, timestep, c_crossattn=None, y=None, **kwargs):
        """
        Apply the diffusion model at a given timestep.
        Compatible with SD3.5 denoiser interface for rectified flow.
        
        Args:
            x: noisy latent (B, C, H, W)
            timestep: rectified flow timestep (B,) - can be float timesteps [0, 1000]
            c_crossattn: cross-attention context (B, L, D)
            y: pooled conditioning (B, D)
            **kwargs: additional arguments
        
        Returns:
            model_output: predicted velocity (for rectified flow) or noise (for diffusion)
        """
        # Handle different timestep formats
        if isinstance(timestep, float):
            timestep = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
        elif torch.is_tensor(timestep):
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).expand(x.shape[0])
            # Convert float timesteps to long if needed
            if timestep.dtype == torch.float32:
                timestep = timestep.long()
        
        # Ensure timestep is in the right format for MMDiTX
        if timestep.dtype != torch.long:
            timestep = timestep.long()
        
        # Use c_crossattn as context if available
        context = c_crossattn
        if context is not None:
            context = self.context_proj(context)
        
        # Apply MMDiTX model
        # Don't pass y since our MMDiTX model doesn't have y_embedder when adm_in_channels is None
        model_output = self.mmdit(x, timestep, context=context)
        
        return model_output
    
    def get_denoised(self, sigma, model_output, model_input):
        """Calculate denoised output using the sampling method."""
        return self.model_sampling.calculate_denoised(sigma, model_output, model_input)
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """Apply noise scaling for the diffusion process."""
        return self.model_sampling.noise_scaling(sigma, noise, latent_image, max_denoise)

    def forward(self, x, timestep, c_crossattn=None, y=None, **kwargs):
        """
        Forward pass for diffusion decoder.
        Compatible with SD3.5 denoiser interface.
        
        Args:
            x: noisy latent tensor (B, C, H, W)
            timestep: diffusion timestep
            c_crossattn: cross-attention context (B, L, D)
            y: pooled conditioning (B, D)
            **kwargs: additional arguments
        
        Returns:
            output: model prediction (B, C, H, W)
        """
        return self.apply_model(x, timestep, c_crossattn=c_crossattn, y=y, **kwargs)

    def _initialize_pos_embed(self):
        """Initialize positional embeddings for the MMDiTX model."""
        if hasattr(self.mmdit, 'pos_embed_max_size') and self.mmdit.pos_embed_max_size is not None:
            from cave.sd35.mmditx import get_2d_sincos_pos_embed
            import numpy as np
            
            # Calculate positional embeddings
            embed_dim = self.mmdit.pos_embed.shape[-1]  # hidden_size
            grid_size = self.mmdit.pos_embed_max_size
            
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_size=grid_size,
                cls_token=False,
                extra_tokens=0,
                scaling_factor=self.mmdit.pos_embed_scaling_factor,
                offset=self.mmdit.pos_embed_offset
            )
            
            # Convert to torch tensor and copy to the buffer
            pos_embed_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.mmdit.pos_embed.copy_(pos_embed_tensor)

class CAVE(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        
        # Initialize encoder and decoder
        self.encoder = CAVEEncoder(config)
        self.decoder = CAVEDecoder(config)
        
        # Initialize diffusion decoder if requested
        self.use_diffusion = config.model.get("use_diffusion", False)
        self.diffusion_decoder = None
        self.diffusion_context_proj = None
        if self.use_diffusion:
            self._init_diffusion_decoder(config)
        
        # Support for text context (optional)
        self.use_text_context = config.model.get("use_text_context", False)
        if self.use_text_context:
            # Text encoder for processing text inputs (e.g., CLIP-like)
            text_embed_dim = config.model.get("text_embed_dim", 768)
            # Use the same hidden size as the encoder
            hidden_size = config.model.encoder.get("depth", 12) * 64
            self.text_proj = nn.Linear(text_embed_dim, hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            if module.weight is not None:
                module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if module.weight is not None:
                module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def encode(self, x, c=None, text_context=None):
        """
        Encode input image to context tokens.
        
        Args:
            x: input image (B, C, H, W)
            c: optional condition (B, D)
            text_context: optional text context (B, L, text_embed_dim)
        
        Returns:
            context: encoded context tokens (B, L, D)
        """
        context = None
        
        # Use text context if provided and model supports it
        if text_context is not None and self.use_text_context:
            # Project text embeddings to model dimension
            context = self.text_proj(text_context)
        
        # Encode using CAVEEncoder
        context = self.encoder(x, c=c, context=context)
        
        return context

    def decode(self, context, x=None, c=None):
        """
        Decode context tokens to output image.
        
        Args:
            context: context tokens from encoder (B, L, D)
            x: noise latent or input image (B, C, H, W). If None, creates noise.
            c: optional condition (B, D)
        
        Returns:
            output: reconstructed/generated image (B, C, H, W)
        """
        batch_size = context.size(0)
        
        if x is None:
            # Create noise latent with proper dimensions
            # For reconstruction decoder, use 3 channels (RGB) and proper input size
            in_channels = 3  # Always 3 for reconstruction (RGB)
            input_size = self.config.model.decoder.get("input_size", 32)  # Patch grid size
            patch_size = self.config.model.decoder.get("patch_size", 8)
            # Calculate the actual pixel size: patch_grid_size * patch_size = image_size
            image_size = input_size * patch_size  # 32 * 8 = 256 pixels
            x = torch.randn(batch_size, in_channels, image_size, image_size, 
                          device=context.device, dtype=context.dtype)
        
        # Decode using CAVEDecoder
        output = self.decoder(x, c=c, context=context)
        
        return output

    def diffusion_decode(self, context, x=None, c=None, steps=20, cfg_scale=7.0, seed=None):
        """
        Decode context tokens to output image using rectified flow sampling.
        
        Args:
            context: context tokens from encoder (B, L, D)
            x: noise latent or input image (B, C, H, W). If None, creates noise.
            c: optional condition (B, D)
            steps: number of rectified flow sampling steps
            cfg_scale: classifier-free guidance scale
            seed: random seed for noise generation
        
        Returns:
            output: reconstructed/generated image (B, C, H, W)
        """
        if not hasattr(self, 'diffusion_decoder') or self.diffusion_decoder is None:
            raise RuntimeError("Diffusion decoder not initialized. Set use_diffusion=True and provide sd3_model_path.")
        
        batch_size = context.size(0)
        device = context.device
        
        # Convert input to latent format if needed
        if x is not None and x.shape[1] == 3:
            # Convert RGB image to 16-channel latent format for diffusion
            x_latent = self._convert_rgb_to_latent(x)
        elif x is None:
            # Create noise latent with proper dimensions for SD3.5 (16 channels, 8x downsampled)
            in_channels = 16  # SD3.5 uses 16-channel latents
            height = self.config.model.get("image_size", 256)
            width = self.config.model.get("image_size", 256)
            latent_h, latent_w = height // 8, width // 8
            
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
                x_latent = torch.randn(batch_size, in_channels, latent_h, latent_w, 
                                      generator=generator, device=device, dtype=context.dtype)
            else:
                x_latent = torch.randn(batch_size, in_channels, latent_h, latent_w, 
                                      device=device, dtype=context.dtype)
        else:
            # Already in latent format
            x_latent = x
        
        # Rectified flow sampling using Euler method
        # Start from noise (t=0) and integrate to data (t=1)
        dt = 1.0 / steps
        
        # Prepare conditioning for CFG
        if c is None:
            # Create dummy conditioning (zeros)
            c = torch.zeros(batch_size, 2048, device=device, dtype=context.dtype)
        
        # For CFG, we need both conditional and unconditional context
        uncond_context = torch.zeros_like(context)
        uncond_c = torch.zeros_like(c)
        
        # Rectified flow sampling loop
        current_x = x_latent.clone()
        
        for i in range(steps):
            t = i * dt  # Current time [0, 1)
            timestep_tensor = torch.full((batch_size,), t * 1000.0, device=device, dtype=torch.float32)
            
            if cfg_scale > 1.0:
                # CFG: predict with both conditional and unconditional
                # Conditional prediction
                cond_velocity = self.diffusion_decoder.apply_model(
                    current_x, timestep_tensor, c_crossattn=context, y=c
                )
                
                # Unconditional prediction
                uncond_velocity = self.diffusion_decoder.apply_model(
                    current_x, timestep_tensor, c_crossattn=uncond_context, y=uncond_c
                )
                
                # Apply classifier-free guidance
                velocity = uncond_velocity + cfg_scale * (cond_velocity - uncond_velocity)
            else:
                # No CFG, just conditional prediction
                velocity = self.diffusion_decoder.apply_model(
                    current_x, timestep_tensor, c_crossattn=context, y=c
                )
            
            # Euler step: x_{t+dt} = x_t + dt * velocity_t
            current_x = current_x + dt * velocity
        
        # Convert latent back to RGB format
        output = self._convert_latent_to_rgb(current_x)
        
        return output
        

    def diffusion_training_step(self, x, context, c=None):
        """
        Perform a single rectified flow training step (SD3.5 style).
        
        In rectified flow, the model learns to predict the velocity field that 
        transforms noise into data along straight paths.
        
        Args:
            x: target image (B, C, H, W)
            context: context tokens from encoder (B, L, D)
            c: optional condition (B, D)
        
        Returns:
            loss: rectified flow training loss
            model_output: predicted velocity
        """
        if self.diffusion_decoder is None:
            raise RuntimeError("Diffusion decoder not available for training.")
        
        batch_size = x.size(0)
        device = x.device
        
        # Convert image to latent space using the same method as diffusion_decode
        x_latent = self._convert_rgb_to_latent(x)
        
        # Sample random timesteps uniformly from [0, 1] for rectified flow
        t = torch.rand(batch_size, device=device, dtype=torch.float32)
        
        # Sample noise (starting point of the flow)
        noise = torch.randn_like(x_latent)
        
        # Rectified flow interpolation: x_t = (1-t) * noise + t * x_latent
        # This creates straight paths from noise to data
        x_t = (1.0 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * x_latent
        
        # The target velocity is the difference: v = x_latent - noise
        # This is the direction from noise to data
        target_velocity = x_latent - noise
        
        # Convert t to timesteps expected by the model (0 to 1000)
        timesteps = t * 1000.0
        
        # Predict the velocity using the diffusion decoder
        predicted_velocity = self.diffusion_decoder.apply_model(
            x_t,
            timesteps,
            c_crossattn=context,
            y=c
        )
        
        # Rectified flow loss: MSE between predicted and target velocity
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        
        return loss, predicted_velocity

    def forward(self, x, c=None, text_context=None, use_diffusion=None, train_diffusion=False, **diffusion_kwargs):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: input image (B, C, H, W)
            c: optional condition (B, D)
            text_context: optional text context (B, L, text_embed_dim)
            use_diffusion: whether to use diffusion decoder (overrides config setting)
            train_diffusion: whether to use diffusion training step instead of sampling
            **diffusion_kwargs: additional arguments for diffusion_decode (steps, cfg_scale, seed)
        
        Returns:
            output: reconstructed image (B, C, H, W)
            extra_results_dict: dictionary with additional outputs (for compatibility with TiTok)
        """
        # Encode to context
        context = self.encode(x, c=c, text_context=text_context)

        # print(f"Context shape: {context.shape}, x shape: {x.shape}, c shape: {c.shape if c is not None else None}, text_context shape: {text_context.shape if text_context is not None else None}")
        
        # Choose decoder based on flag
        should_use_diffusion = use_diffusion if use_diffusion is not None else self.use_diffusion
        
        # Initialize extra results dictionary
        extra_results_dict = {
            "context": context,
            "quantizer_loss": torch.tensor(0.0, device=x.device),
            "commitment_loss": torch.tensor(0.0, device=x.device),
            "codebook_loss": torch.tensor(0.0, device=x.device),
            "rectified_flow_loss": torch.tensor(0.0, device=x.device),  # Updated name
            "used_diffusion": should_use_diffusion and self.diffusion_decoder is not None,
        }
        
        if should_use_diffusion and self.diffusion_decoder is not None:
            if train_diffusion and self.training:
                # During training, use rectified flow training step
                rectified_flow_loss, predicted_velocity = self.diffusion_training_step(x, context, c=c)
                extra_results_dict["rectified_flow_loss"] = rectified_flow_loss
                
                # For training, we still need to return a reconstructed image
                # Use the reconstruction decoder for this (no input x)
                output = self.decode(context, x=None, c=c)
            else:
                # During inference, use rectified flow sampling
                output = self.diffusion_decode(context, x=None, c=c, **diffusion_kwargs)
        else:
            # Use reconstruction decoder (no input x - reconstruct from context only)
            output = self.decode(context, x=None, c=c)
        
        # print(f"Output shape: {output.shape}, input x shape: {x.shape}")
        
        return output, extra_results_dict

    def encode_tokens(self, x, c=None, text_context=None):
        """
        Encode image to discrete tokens (for compatibility).
        
        Args:
            x: input image (B, C, H, W)
            c: optional condition (B, D)
            text_context: optional text context (B, L, text_embed_dim)
        
        Returns:
            tokens: encoded context tokens (B, L, D)
        """
        return self.encode(x, c=c, text_context=text_context)

    def decode_tokens(self, tokens, c=None, use_diffusion=None, **diffusion_kwargs):
        """
        Decode tokens to image (for compatibility).
        
        Args:
            tokens: context tokens (B, L, D)
            c: optional condition (B, D)
            use_diffusion: whether to use diffusion decoder (overrides config setting)
            **diffusion_kwargs: additional arguments for diffusion_decode
        
        Returns:
            output: reconstructed image (B, C, H, W)
        """
        should_use_diffusion = use_diffusion if use_diffusion is not None else self.use_diffusion
        
        if should_use_diffusion and self.diffusion_decoder is not None:
            return self.diffusion_decode(tokens, c=c, **diffusion_kwargs)
        else:
            return self.decode(tokens, c=c)
    
    def set_diffusion_mode(self, use_diffusion):
        """
        Set whether to use diffusion decoder by default.
        
        Args:
            use_diffusion: boolean flag to enable/disable diffusion decoder
        """
        if use_diffusion and self.diffusion_decoder is None:
            raise RuntimeError("Diffusion decoder not available. Initialize with use_diffusion=True and sd3_model_path.")
        self.use_diffusion = use_diffusion
    
    def generate_from_context(self, context, c=None, steps=20, cfg_scale=7.0, seed=None):
        """
        Generate image from context using diffusion decoder.
        
        Args:
            context: context tokens from encoder (B, L, D)
            c: optional condition (B, D)
            steps: number of diffusion steps
            cfg_scale: classifier-free guidance scale
            seed: random seed
        
        Returns:
            output: generated image (B, C, H, W)
        """
        if self.diffusion_decoder is None:
            raise RuntimeError("Diffusion decoder not available. Initialize with use_diffusion=True and sd3_model_path.")
        
        return self.diffusion_decode(context, c=c, steps=steps, cfg_scale=cfg_scale, seed=seed)

    def _init_diffusion_decoder(self, config):
        """Initialize diffusion decoder with pretrained SD3.5 weights."""
        try:
            from cave.sd35.sd3_impls import BaseModel as SD3BaseModel, ModelSamplingDiscreteFlow
            from safetensors import safe_open
            import os
            
            # Get SD3.5 model path from config
            sd3_model_path = config.model.get("sd3_model_path", None)
            if sd3_model_path is None or not os.path.exists(sd3_model_path):
                print("Warning: SD3.5 model path not found. Diffusion decoder will use random weights.")
                self.diffusion_decoder = CAVEDiffusionDecoder(config)
                return
            
            print(f"Loading SD3.5 pretrained weights from: {sd3_model_path}")
            
            # Create CAVE diffusion decoder first
            self.diffusion_decoder = CAVEDiffusionDecoder(config)
            
            # Load the SD3.5 pretrained weights into the MMDiTX component
            with safe_open(sd3_model_path, framework="pt", device="cpu") as f:
                self._load_into_diffusion_model(f, self.diffusion_decoder.mmdit, "model.diffusion_model.")
            
            print("SD3.5 diffusion decoder loaded successfully!")
            
        except Exception as e:
            print(f"Error loading SD3.5 diffusion decoder: {e}")
            print("Falling back to CAVEDiffusionDecoder with random weights.")
            self.diffusion_decoder = CAVEDiffusionDecoder(config)
    
    def _load_into_diffusion_model(self, ckpt, model, prefix):
        """Load weights from safetensors file into the MMDiTX diffusion model."""
        loaded_keys = []
        skipped_keys = []
        
        for key in ckpt.keys():
            if key.startswith(prefix) and not key.startswith("loss."):
                model_key = key[len(prefix):]
                
                # Skip keys that don't exist in our model structure
                if any(skip_pattern in model_key for skip_pattern in [
                    "y_embedder",  # We don't use y_embedder in our CAVE setup
                    "context_embedder"  # We handle context differently
                ]):
                    skipped_keys.append(key)
                    continue
                
                # Navigate to the correct parameter
                path = model_key.split(".")
                obj = model
                
                for p in path:
                    if hasattr(obj, p):
                        obj = getattr(obj, p)
                    else:
                        skipped_keys.append(key)
                        obj = None
                        break
                
                if obj is not None and hasattr(obj, 'data'):
                    try:
                        tensor = ckpt.get_tensor(key)
                        if tensor.shape == obj.shape:
                            obj.data.copy_(tensor)
                            loaded_keys.append(key)
                        else:
                            print(f"Shape mismatch for {key}: expected {obj.shape}, got {tensor.shape}")
                            skipped_keys.append(key)
                    except Exception as e:
                        print(f"Error loading {key}: {e}")
                        skipped_keys.append(key)
        
        print(f"Loaded {len(loaded_keys)} parameters, skipped {len(skipped_keys)} parameters")
        if len(skipped_keys) > 0:
            print(f"Skipped keys: {skipped_keys[:10]}..." if len(skipped_keys) > 10 else f"Skipped keys: {skipped_keys}")
    
    def _get_sigmas(self, steps):
        """Generate sigma schedule for diffusion sampling."""
        if hasattr(self.diffusion_decoder, 'model_sampling'):
            sampling = self.diffusion_decoder.model_sampling
        else:
            from cave.sd35.sd3_impls import ModelSamplingDiscreteFlow
            sampling = ModelSamplingDiscreteFlow(shift=1.0)
            
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)
    
    def _convert_rgb_to_latent(self, x):
        """
        Convert RGB image to 16-channel latent format for diffusion.
        This is a simplified conversion - in practice you'd use a VAE encoder.
        
        Args:
            x: RGB image (B, 3, H, W)
        
        Returns:
            latent: 16-channel latent (B, 16, H/8, W/8)
        """
        # Downsample to latent size (8x downsampling like SD3.5)
        x_downsampled = torch.nn.functional.interpolate(
            x, size=(x.shape[2] // 8, x.shape[3] // 8), 
            mode='bilinear', align_corners=False
        )
        
        # Expand to 16 channels by repeating and adding variations
        if x_downsampled.shape[1] != 16:
            # Repeat channels and add small random variations
            repeat_factor = 16 // x_downsampled.shape[1]
            remainder = 16 % x_downsampled.shape[1]
            
            repeated = x_downsampled.repeat(1, repeat_factor, 1, 1)
            if remainder > 0:
                extra = x_downsampled[:, :remainder, :, :]
                x_latent = torch.cat([repeated, extra], dim=1)
            else:
                x_latent = repeated
            
            # Add small noise to make channels slightly different
            x_latent = x_latent + 0.01 * torch.randn_like(x_latent)
        else:
            x_latent = x_downsampled
        
        return x_latent
    
    def _convert_latent_to_rgb(self, x_latent):
        """
        Convert 16-channel latent back to RGB format.
        This is a simplified conversion - in practice you'd use a VAE decoder.
        
        Args:
            x_latent: 16-channel latent (B, 16, H, W)
        
        Returns:
            rgb: RGB image (B, 3, H*8, W*8)
        """
        # Take the first 3 channels as RGB approximation
        if x_latent.shape[1] >= 3:
            rgb_latent = x_latent[:, :3, :, :]
        else:
            # Repeat channels if we have fewer than 3
            rgb_latent = x_latent.repeat(1, 3 // x_latent.shape[1] + 1, 1, 1)[:, :3, :, :]
        
        # Upsample back to original size (8x upsampling)
        target_h = rgb_latent.shape[2] * 8
        target_w = rgb_latent.shape[3] * 8
        
        rgb = torch.nn.functional.interpolate(
            rgb_latent, size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )
        
        # Clamp to valid range
        rgb = torch.clamp(rgb, -1.0, 1.0)
        
        return rgb
