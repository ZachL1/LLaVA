import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from omegaconf import OmegaConf
from cave.cave_with_vae import CAVEWithVAE

# Import dual_vision modules
try:
    from dual_vision import DualCLIPVisionEncoder, DualVisionConfig
    DUAL_VISION_AVAILABLE = True
except ImportError:
    DUAL_VISION_AVAILABLE = False
    print("Warning: dual_vision module not found. DualVisionTower will not be available.")

class DualVisionTower(nn.Module):
    """Dual Vision Tower wrapper for LLaVA integration.
    
    Supports dual-branch vision encoder with MoT architecture.
    Compatible with CLIP, ViT, and SigLIP base models.
    
    The right branch can use either random tokens or LLM hidden states as input.
    When text_hidden_states are provided to forward(), they are projected to vision
    hidden dimension and used as auxiliary tokens for the right branch.
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        
        if not DUAL_VISION_AVAILABLE:
            raise ImportError(
                "dual_vision module is required but not found. "
                "Please ensure dual_vision package is in the Python path."
            )
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        
        # Parse vision_tower string for encoder type and model path
        # Format: "dual_vision_{encoder_type}_{model_path}"
        # Examples: "dual_vision_clip_openai/clip-vit-large-patch14-336"
        #           "dual_vision_siglip_google/siglip-so400m-patch14-384"
        
        # Remove "dual_vision_" prefix first
        if vision_tower.startswith("dual_vision_"):
            remainder = vision_tower[len("dual_vision_"):]  # e.g., "clip_openai/clip-vit-large-patch14-336"
            # Split once to get encoder_type and model_path
            parts = remainder.split('_', 1)
            if len(parts) >= 2:
                self.encoder_type = parts[0]  # clip, siglip, or vit
                self.pretrained_path = parts[1]  # e.g., "openai/clip-vit-large-patch14-336"
            elif len(parts) == 1:
                # Only encoder type provided, no underscore in remainder
                self.encoder_type = parts[0]
                self.pretrained_path = getattr(args, 'dual_vision_pretrained', None)
            else:
                # Fallback
                self.encoder_type = 'clip'
                self.pretrained_path = getattr(args, 'dual_vision_pretrained', 'openai/clip-vit-large-patch14-336')
        else:
            # No proper prefix, use defaults
            self.encoder_type = 'clip'
            self.pretrained_path = getattr(args, 'dual_vision_pretrained', 'openai/clip-vit-large-patch14-336')
        
        # Get dual_vision specific args
        self.num_auxiliary_tokens = getattr(args, 'dual_vision_num_aux_tokens', None)
        self.output_mode = getattr(args, 'dual_vision_output_mode', 'right')  # 'right', 'left', 'both', 'concat'
        self.train_right_branch = getattr(args, 'dual_vision_train_right', False)
        self.use_flash_attention = getattr(args, 'dual_vision_flash_attn', True)
        self.attention_mode = getattr(args, 'dual_vision_attention_mode', 'joint')  # 'joint' or 'cross'
        self.adaptation_checkpoint = getattr(args, 'dual_vision_adaptation_ckpt', None)  # Path to adaptation trained weights
        
        # LLM hidden state projection layer
        self.llm_hidden_proj = None
        self.llm_hidden_size = None
        
        # Create vision_tower and image_processor in __init__ (like CAVEEncoderTower)
        # This ensures they are registered as submodules and handled by HuggingFace device_map
        self._create_vision_tower_and_processor()
        
        # Initialize llm_hidden_proj in __init__ so HuggingFace can load weights automatically
        # args is the model config which contains hidden_size (LLM hidden dimension)
        llm_hidden_size = getattr(args, 'hidden_size', None)
        if llm_hidden_size is not None:
            self.set_llm_hidden_size(llm_hidden_size)
        
        # Load weights if not delay_load
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
    
    def _create_vision_tower_and_processor(self):
        """Create vision_tower and image_processor based on encoder_type.
        
        This is called in __init__ to ensure vision_tower is a registered submodule,
        allowing HuggingFace's device_map to handle device placement automatically.
        """
        print(f"Creating Dual Vision Tower: encoder_type={self.encoder_type}, pretrained={self.pretrained_path}")
        
        if self.encoder_type == 'clip':
            # Load CLIP config
            clip_config = CLIPVisionConfig.from_pretrained(self.pretrained_path)
            
            # Create DualVisionConfig from CLIP config
            config = DualVisionConfig(
                encoder_type='clip',
                hidden_size=clip_config.hidden_size,
                num_layers=clip_config.num_hidden_layers,
                num_heads=clip_config.num_attention_heads,
                intermediate_size=clip_config.intermediate_size,
                image_size=clip_config.image_size,
                patch_size=clip_config.patch_size,
                num_channels=clip_config.num_channels,
                layer_norm_eps=clip_config.layer_norm_eps,
                output_mode=self.output_mode,
                freeze_left_branch=True,
                use_flash_attention=self.use_flash_attention,
                attention_mode=self.attention_mode,
            )
            
            # Create dual vision encoder
            self.vision_tower = DualCLIPVisionEncoder(config)
            
            # Get image processor
            self.image_processor = CLIPImageProcessor.from_pretrained(self.pretrained_path)
            
            # Store config for later use
            self._config = config
            
        elif self.encoder_type == 'siglip':
            try:
                from transformers import SiglipVisionConfig, SiglipImageProcessor
                from dual_vision import DualSigLIPVisionEncoder
                
                # Load SigLIP config
                siglip_config = SiglipVisionConfig.from_pretrained(self.pretrained_path)
                
                # Create DualVisionConfig
                config = DualVisionConfig(
                    encoder_type='siglip',
                    hidden_size=siglip_config.hidden_size,
                    num_layers=siglip_config.num_hidden_layers,
                    num_heads=siglip_config.num_attention_heads,
                    intermediate_size=siglip_config.intermediate_size,
                    image_size=siglip_config.image_size,
                    patch_size=siglip_config.patch_size,
                    num_channels=siglip_config.num_channels,
                    layer_norm_eps=siglip_config.layer_norm_eps,
                    output_mode=self.output_mode,
                    freeze_left_branch=True,
                    use_flash_attention=self.use_flash_attention,
                    attention_mode=self.attention_mode,
                )
                
                # Create dual vision encoder
                self.vision_tower = DualSigLIPVisionEncoder(config)
                
                # Get image processor
                self.image_processor = SiglipImageProcessor.from_pretrained(self.pretrained_path)
                
                self._config = config
                
            except ImportError:
                raise ImportError("SigLIP support requires transformers >= 4.30.0")
        
        elif self.encoder_type == 'vit':
            try:
                from transformers import ViTConfig, ViTImageProcessor
                from dual_vision import DualViTEncoder
                
                # Load ViT config
                vit_config = ViTConfig.from_pretrained(self.pretrained_path)
                
                # Create DualVisionConfig
                config = DualVisionConfig(
                    encoder_type='vit',
                    hidden_size=vit_config.hidden_size,
                    num_layers=vit_config.num_hidden_layers,
                    num_heads=vit_config.num_attention_heads,
                    intermediate_size=vit_config.intermediate_size,
                    image_size=vit_config.image_size,
                    patch_size=vit_config.patch_size,
                    num_channels=vit_config.num_channels,
                    layer_norm_eps=vit_config.layer_norm_eps,
                    output_mode=self.output_mode,
                    freeze_left_branch=True,
                    use_flash_attention=self.use_flash_attention,
                    attention_mode=self.attention_mode,
                )
                
                # Create dual vision encoder
                self.vision_tower = DualViTEncoder(config)
                
                # Get image processor
                self.image_processor = ViTImageProcessor.from_pretrained(self.pretrained_path)
                
                self._config = config
                
            except ImportError:
                raise ImportError("ViT support requires transformers library")
        
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        
        print(f"Dual Vision Tower structure created: {self.vision_tower_name}")
    
    def load_model(self, device_map=None):
        """Load pretrained weights into the vision_tower.
        
        Note: vision_tower is already created in __init__, this only loads weights.
        """
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        print(f"Loading weights for Dual Vision Tower: {self.vision_tower_name}")
        
        # Set requires_grad based on configuration
        if not self.train_right_branch:
            # Don't freeze by default (let LLaVA training script decide)
            self.vision_tower.requires_grad_(False)
        else:
            self.vision_tower.freeze_left_branch()
        
        # Load adaptation checkpoint if provided
        if self.adaptation_checkpoint is not None:
            self.load_adaptation_weights(self.adaptation_checkpoint)
        else:
            # Load pretrained weights
            self.vision_tower.load_pretrained_weights(self.pretrained_path, strict=False)
        
        self.is_loaded = True
        print(f"Dual Vision Tower weights loaded successfully: {self.vision_tower_name}")
    
    def load_adaptation_weights(self, checkpoint_path):
        """Load weights from adaptation training checkpoint.
        
        Args:
            checkpoint_path: Path to adaptation checkpoint (.pt file)
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Adaptation checkpoint not found: {checkpoint_path}")
        
        print(f"Loading adaptation weights from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # If it's a full training checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'epoch' in checkpoint:
                    print(f"  - Loaded from epoch {checkpoint['epoch']}")
                if 'train_loss' in checkpoint:
                    print(f"  - Training loss: {checkpoint['train_loss']:.4f}")
            # If it's just a state dict
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format")
        
        # Load the state dict into vision tower
        # The adaptation training saves the full model state dict
        missing_keys, unexpected_keys = self.vision_tower.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  - Missing keys (expected for partial load): {len(missing_keys)}")
            # Only show first few for brevity
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    - {key}")
            else:
                print(f"    - {missing_keys[:5]}...")
                print(f"    - ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"  - Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                print(f"    - {unexpected_keys[:5]}...")
                print(f"    - ... and {len(unexpected_keys) - 5} more")
        
        # Count loaded parameters
        loaded_params = len(state_dict)
        right_branch_params = len([k for k in state_dict.keys() if '_mot' in k])
        left_branch_params = loaded_params - right_branch_params
        
        print(f"  - Loaded {loaded_params} parameters total")
        print(f"    - Left branch: {left_branch_params} parameters")
        print(f"    - Right branch: {right_branch_params} parameters")
        print("✓ Adaptation weights loaded successfully")
    
    def set_llm_hidden_size(self, llm_hidden_size):
        """Initialize the projection layer for LLM hidden states.
        
        This must be called after the model is loaded and before using
        text_hidden_states in forward().
        
        Args:
            llm_hidden_size: Hidden dimension of the LLM (e.g., 4096 for LLaMA-7B)
        """
        self.llm_hidden_size = llm_hidden_size
        vision_hidden_size = self._config.hidden_size
        
        # Simple linear projection: LLM hidden dim -> Vision hidden dim
        self.llm_hidden_proj = nn.Linear(llm_hidden_size, vision_hidden_size)
        
        # Initialize with small weights
        nn.init.normal_(self.llm_hidden_proj.weight, std=0.02)
        nn.init.zeros_(self.llm_hidden_proj.bias)
        
        # Move to same device and dtype as vision tower
        self.llm_hidden_proj = self.llm_hidden_proj.to(
            device=self.device,
            dtype=self.dtype
        )
        
        print(f"Initialized LLM hidden projection: {llm_hidden_size} -> {vision_hidden_size}")
    
    def _align_sequence_length(self, hidden_states, target_length):
        """Align hidden states sequence length to target length via truncation/padding.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_dim]
            target_length: Target sequence length
            
        Returns:
            Tensor of shape [batch_size, target_length, hidden_dim]
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        if seq_length == target_length:
            return hidden_states
        elif seq_length > target_length:
            # Truncate: keep the last target_length tokens (more recent context)
            return hidden_states[:, -target_length:, :]
        else:
            # Pad: replicate the last token to fill
            padding_length = target_length - seq_length
            last_token = hidden_states[:, -1:, :].expand(-1, padding_length, -1)
            return torch.cat([hidden_states, last_token], dim=1)
    
    def _get_learnable_auxiliary_tokens(self, batch_size, device, dtype):
        """Get learnable auxiliary tokens from the vision tower.
        
        Args:
            batch_size: Batch size for expansion
            device: Target device
            dtype: Target dtype
            
        Returns:
            Learnable auxiliary tokens [batch_size, num_patches, hidden_dim]
        """
        if hasattr(self.vision_tower, 'learnable_auxiliary_tokens'):
            # Expand learnable tokens to batch size
            learnable_tokens = self.vision_tower.learnable_auxiliary_tokens.expand(batch_size, -1, -1)
            return learnable_tokens.to(device=device, dtype=dtype)
        else:
            # Fallback to zeros if learnable tokens not available
            return torch.zeros(
                batch_size, self._config.num_patches, self._config.hidden_size,
                device=device, dtype=dtype
            )
    
    @torch.no_grad()
    def forward(self, images, text_hidden_states=None):
        """Forward pass through dual vision tower.
        
        Args:
            images: Image tensor [batch_size, channels, height, width] or list of tensors
            text_hidden_states: Optional LLM hidden states for right branch input.
                - If provided as a tensor: [batch_size, seq_length, llm_hidden_dim]
                - If provided as a list: list of [1, seq_length, llm_hidden_dim] tensors
                - If None: uses learnable tokens only (from the encoder)
                - When provided: learnable tokens + projected text hidden states
            
        Returns:
            Image features based on output_mode setting
        """
        batch_size = images.shape[0] if not isinstance(images, list) else len(images)
        
        # Determine number of auxiliary tokens
        if self.num_auxiliary_tokens is None:
            # Default: same as number of image patches
            num_aux_tokens = self._config.num_patches
        else:
            num_aux_tokens = self.num_auxiliary_tokens
        
        # Check if we can use text hidden states
        use_text_hidden_states = (
            text_hidden_states is not None and 
            self.llm_hidden_proj is not None
        )
        
        if type(images) is list:
            # Handle list of images
            auxiliary_tokens_list = []
            
            for idx, image in enumerate(images):
                # Start with learnable tokens as base
                learnable_tokens = self._get_learnable_auxiliary_tokens(
                    batch_size=1, 
                    device=image.device, 
                    dtype=image.dtype
                )  # [1, num_patches, hidden_dim]
                
                if use_text_hidden_states:
                    # Get text hidden states for this image
                    if isinstance(text_hidden_states, list):
                        text_hs = text_hidden_states[idx]  # [1, seq_len, llm_hidden_dim]
                    else:
                        text_hs = text_hidden_states[idx:idx+1]  # [1, seq_len, llm_hidden_dim]
                    
                    # Project to vision hidden dimension
                    projected_hs = self.llm_hidden_proj(text_hs)  # [1, seq_len, vision_hidden_dim]
                    
                    # Align sequence length to match num_aux_tokens
                    projected_hs = self._align_sequence_length(projected_hs, num_aux_tokens)
                    projected_hs = projected_hs.to(device=image.device, dtype=image.dtype)
                    
                    # Combine: learnable tokens + projected text hidden states
                    aux_tokens = learnable_tokens + projected_hs
                else:
                    # Use learnable tokens only (encoder will use them as default)
                    aux_tokens = None  # Let encoder use its learnable tokens
                
                auxiliary_tokens_list.append(aux_tokens)
            
            # Process each image individually
            image_features = []
            for image, aux_tokens in zip(images, auxiliary_tokens_list):
                # Forward through dual encoder
                output = self.vision_tower(
                    image.unsqueeze(0),
                    auxiliary_tokens=aux_tokens
                )
                
                # Extract features based on output mode
                if self.output_mode == 'right':
                    features = output['right_last_hidden_state']
                elif self.output_mode == 'left':
                    features = output['left_last_hidden_state']
                elif self.output_mode == 'concat':
                    # Concatenate left and right
                    features = torch.cat([output['left_last_hidden_state'], output['right_last_hidden_state']], dim=-1)
                elif self.output_mode == 'both':
                    # Return both, but for compatibility return right as default
                    features = output['right_last_hidden_state']
                else:
                    features = output['right_last_hidden_state']
                
                # Remove CLS token if present and not needed
                if self._config.use_cls_token and features.shape[1] > self._config.num_patches:
                    features = features[:, 1:]  # Remove CLS token, keep only patches
                
                image_features.append(features)
        else:
            # Batch processing
            if use_text_hidden_states:
                # Start with learnable tokens as base
                learnable_tokens = self._get_learnable_auxiliary_tokens(
                    batch_size=batch_size, 
                    device=images.device, 
                    dtype=images.dtype
                )  # [batch_size, num_patches, hidden_dim]
                
                # Project text hidden states to vision hidden dimension
                # text_hidden_states: [batch_size, seq_len, llm_hidden_dim]
                projected_hs = self.llm_hidden_proj(text_hidden_states)  # [batch_size, seq_len, vision_hidden_dim]
                
                # Align sequence length to match num_aux_tokens
                projected_hs = self._align_sequence_length(projected_hs, num_aux_tokens)
                projected_hs = projected_hs.to(device=images.device, dtype=images.dtype)
                
                # Combine: learnable tokens + projected text hidden states
                auxiliary_tokens = learnable_tokens + projected_hs
            else:
                # Use learnable tokens only (encoder will use them as default)
                auxiliary_tokens = None  # Let encoder use its learnable tokens
            
            # Forward through dual encoder
            output = self.vision_tower(
                images,
                auxiliary_tokens=auxiliary_tokens
            )
            
            # Extract features based on output mode
            if self.output_mode == 'right':
                image_features = output['right_last_hidden_state']
            elif self.output_mode == 'left':
                image_features = output['left_last_hidden_state']
            elif self.output_mode == 'concat':
                # Concatenate left and right along feature dimension
                image_features = torch.cat([output['left_last_hidden_state'], output['right_last_hidden_state']], dim=-1)
            elif self.output_mode == 'both':
                image_features = output['right_last_hidden_state']
            else:
                image_features = output['right_last_hidden_state']
            
            # Remove CLS token if present and not needed
            if self._config.use_cls_token and image_features.shape[1] > self._config.num_patches:
                image_features = image_features[:, 1:]
        
        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype
    
    @property
    def device(self):
        return next(self.vision_tower.parameters()).device
    
    @property
    def config(self):
        return self._config
    
    @property
    def hidden_size(self):
        if self.output_mode == 'concat':
            # Concatenated output has 2x hidden size
            return self._config.hidden_size * 2
        return self._config.hidden_size
    
    @property
    def num_patches_per_side(self):
        return self._config.image_size // self._config.patch_size
    
    @property
    def num_patches(self):
        return self._config.num_patches


class CAVEEncoderTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.use_learnable = "learnable" in vision_tower
        self.kl = "kl" in vision_tower
        self.vision_tower = CAVEWithVAE(OmegaConf.load(args.cave_config))
        self.cave_ckpt = args.cave_ckpt
        self.cave_token = args.cave_token
        # {
        #     "crop_size": 256,
        #     "do_center_crop": true,
        #     "do_normalize": true,
        #     "do_resize": true,
        #     "feature_extractor_type": "CLIPFeatureExtractor",
        #     "image_mean": [
        #         0.5,
        #         0.5,
        #         0.5
        #     ],
        #     "image_std": [
        #         0.5,
        #         0.5,
        #         0.5
        #     ],
        #     "resample": 3,
        #     "size": 256
        # }
        self.image_processor = CLIPImageProcessor(
            crop_size=256,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            resample=3,
            size=256,
        )
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.cave_ckpt))
            return

        self.vision_tower.load_checkpoint(self.cave_ckpt, kl=self.kl, use_learnable=self.use_learnable)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.is_loaded = True
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower.encode(image.unsqueeze(0), num_context_tokens=self.cave_token, use_vae_training=self.kl)
                image_features.append(image_feature)
        else:
            image_feature = self.vision_tower.encode(images, num_context_tokens=self.cave_token, use_vae_training=self.kl)
        
        return image_feature

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1536

    @property
    def num_patches(self):
        return  self.cave_token


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs: BaseModelOutputWithPooling):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs: BaseModelOutputWithPooling = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
