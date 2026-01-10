import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from omegaconf import OmegaConf
from cave.cave_with_vae import CAVEWithVAE

# Import dual_vision modules
try:
    from dual_vision import DualCLIPVisionEncoder, DualVisionConfig
    from dual_vision.utils import generate_random_tokens
    DUAL_VISION_AVAILABLE = True
except ImportError:
    DUAL_VISION_AVAILABLE = False
    print("Warning: dual_vision module not found. DualVisionTower will not be available.")

class DualVisionTower(nn.Module):
    """Dual Vision Tower wrapper for LLaVA integration.
    
    Supports dual-branch vision encoder with MoT architecture.
    Compatible with CLIP, ViT, and SigLIP base models.
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
        
        # Image processor will be set when loading model
        self.image_processor = None
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        print(f"Loading Dual Vision Tower: encoder_type={self.encoder_type}, pretrained={self.pretrained_path}")
        
        # Load pretrained model to get config
        from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor
        
        if self.encoder_type == 'clip':
            # Load CLIP config
            clip_model = CLIPVisionModel.from_pretrained(self.pretrained_path)
            clip_config = clip_model.config
            
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
                from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
                from dual_vision import DualSigLIPVisionEncoder
                
                # Load SigLIP config
                siglip_model = SiglipVisionModel.from_pretrained(self.pretrained_path)
                siglip_config = siglip_model.config
                
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
                from transformers import ViTModel, ViTConfig, ViTImageProcessor
                from dual_vision import DualViTEncoder
                
                # Load ViT config
                vit_model = ViTModel.from_pretrained(self.pretrained_path)
                vit_config = vit_model.config
                
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
        print(f"Dual Vision Tower loaded successfully: {self.vision_tower_name}")
    
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
    
    @torch.no_grad()
    def forward(self, images):
        """Forward pass through dual vision tower.
        
        Args:
            images: Image tensor [batch_size, channels, height, width]
            
        Returns:
            Image features based on output_mode setting
        """
        batch_size = images.shape[0] if not isinstance(images, list) else len(images)
        
        # Determine number of auxiliary tokens
        if self.num_auxiliary_tokens is None:
            # Default: same as number of image patches
            num_aux_tokens = self._config.num_patches
            if self._config.use_cls_token:
                num_aux_tokens += 1
        else:
            num_aux_tokens = self.num_auxiliary_tokens
        
        # Generate random auxiliary tokens for right branch
        # This will be replaced with more meaningful tokens in future versions
        # (e.g., text embeddings, LLM hidden states, etc.)
        if type(images) is list:
            # Handle list of images
            auxiliary_tokens_list = []
            for image in images:
                aux_tokens = generate_random_tokens(
                    batch_size=1,
                    seq_length=num_aux_tokens,
                    hidden_dim=self._config.hidden_size,
                    distribution='gaussian',
                    std=0.02,
                    device=image.device,
                    dtype=image.dtype
                )
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
            auxiliary_tokens = generate_random_tokens(
                batch_size=batch_size,
                seq_length=num_aux_tokens,
                hidden_dim=self._config.hidden_size,
                distribution='gaussian',
                std=0.02,
                device=images.device,
                dtype=images.dtype
            )
            
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
