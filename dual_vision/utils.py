"""Utility functions for Dual Vision Encoder"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import warnings


def pad_auxiliary_tokens(
    auxiliary_tokens: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0
) -> torch.Tensor:
    """Pad or truncate auxiliary tokens to target length.
    
    Args:
        auxiliary_tokens: Input tokens [batch_size, seq_len, hidden_dim]
        target_length: Target sequence length
        pad_value: Value to use for padding
        
    Returns:
        Padded/truncated tokens [batch_size, target_length, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = auxiliary_tokens.shape
    
    if seq_len == target_length:
        return auxiliary_tokens
    elif seq_len < target_length:
        # Pad
        padding = torch.full(
            (batch_size, target_length - seq_len, hidden_dim),
            pad_value,
            dtype=auxiliary_tokens.dtype,
            device=auxiliary_tokens.device
        )
        return torch.cat([auxiliary_tokens, padding], dim=1)
    else:
        # Truncate
        warnings.warn(
            f"Truncating auxiliary tokens from {seq_len} to {target_length}. "
            "This may result in information loss."
        )
        return auxiliary_tokens[:, :target_length, :]


def generate_random_tokens(
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    distribution: str = 'gaussian',
    std: float = 0.02,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Generate random tokens for the right branch.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_dim: Hidden dimension
        distribution: 'gaussian' or 'uniform'
        std: Standard deviation for Gaussian distribution
        device: Device to create tokens on
        dtype: Data type for tokens
        
    Returns:
        Random tokens [batch_size, seq_length, hidden_dim]
    """
    if distribution == 'gaussian':
        tokens = torch.randn(
            batch_size, seq_length, hidden_dim,
            device=device, dtype=dtype
        ) * std
    elif distribution == 'uniform':
        # Uniform in [-std, std]
        tokens = torch.rand(
            batch_size, seq_length, hidden_dim,
            device=device, dtype=dtype
        ) * 2 * std - std
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return tokens


def rename_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    add_suffix: Optional[str] = None,
    remove_suffix: Optional[str] = None,
    key_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """Rename keys in state dict by adding/removing suffix or using custom mapping.
    
    Args:
        state_dict: Original state dict
        add_suffix: Suffix to add to all keys (e.g., '_mot')
        remove_suffix: Suffix to remove from all keys
        key_mapping: Custom key mapping dictionary
        
    Returns:
        State dict with renamed keys
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Apply custom mapping first
        if key_mapping and key in key_mapping:
            new_key = key_mapping[key]
        else:
            # Remove suffix
            if remove_suffix and new_key.endswith(remove_suffix):
                new_key = new_key[:-len(remove_suffix)]
            
            # Add suffix
            if add_suffix:
                new_key = new_key + add_suffix
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def get_parameter_groups(
    model: nn.Module,
    left_branch_lr: float = 1e-5,
    right_branch_lr: float = 1e-4,
    other_lr: float = 1e-4
) -> List[Dict]:
    """Create parameter groups for optimizer with different learning rates.
    
    Args:
        model: Dual vision encoder model
        left_branch_lr: Learning rate for left branch (image) parameters
        right_branch_lr: Learning rate for right branch (MoT) parameters
        other_lr: Learning rate for other parameters
        
    Returns:
        List of parameter groups for optimizer
    """
    left_params = []
    right_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if '_mot' in name:
            right_params.append(param)
        elif any(x in name for x in ['layers', 'attention', 'mlp', 'norm']):
            # Left branch transformer parameters (without _mot suffix)
            left_params.append(param)
        else:
            # Other parameters (embeddings, etc.)
            other_params.append(param)
    
    param_groups = []
    if left_params:
        param_groups.append({'params': left_params, 'lr': left_branch_lr})
    if right_params:
        param_groups.append({'params': right_params, 'lr': right_branch_lr})
    if other_params:
        param_groups.append({'params': other_params, 'lr': other_lr})
    
    return param_groups


def load_vision_encoder_weights(
    checkpoint_path: str,
    encoder_type: str = 'auto'
) -> Dict[str, torch.Tensor]:
    """Load weights from a vision encoder checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint or HuggingFace model name
        encoder_type: Type of encoder ('auto', 'vit', 'clip', 'siglip')
        
    Returns:
        State dict with loaded weights
    """
    # Try to load from HuggingFace first
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(checkpoint_path)
        state_dict = model.state_dict()
        
        # Auto-detect encoder type if needed
        if encoder_type == 'auto':
            model_class = model.__class__.__name__.lower()
            if 'clip' in model_class:
                encoder_type = 'clip'
            elif 'siglip' in model_class:
                encoder_type = 'siglip'
            else:
                encoder_type = 'vit'
        
        return state_dict
    except:
        # Fall back to local checkpoint
        import os
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            return state_dict
        else:
            raise ValueError(f"Could not load checkpoint from {checkpoint_path}")


def create_dual_encoder(
    encoder_type: str,
    config: Optional[object] = None,
    pretrained_path: Optional[str] = None,
    **kwargs
):
    """Factory function to create a dual vision encoder.
    
    Args:
        encoder_type: Type of encoder ('vit', 'clip', 'siglip')
        config: DualVisionConfig object (optional)
        pretrained_path: Path to pretrained weights
        **kwargs: Additional arguments to override in config
        
    Returns:
        Initialized dual vision encoder
    """
    from config import DualVisionConfig
    
    # Create config if not provided
    if config is None:
        # If pretrained_path is provided, extract config from pretrained model
        if pretrained_path:
            config = _create_config_from_pretrained(encoder_type, pretrained_path, **kwargs)
        else:
            config = DualVisionConfig(encoder_type=encoder_type, **kwargs)
    
    # Import appropriate encoder class
    if encoder_type == 'vit':
        from modeling_vit import DualViTEncoder
        encoder = DualViTEncoder(config)
    elif encoder_type == 'clip':
        from modeling_clip import DualCLIPVisionEncoder
        encoder = DualCLIPVisionEncoder(config)
    elif encoder_type == 'siglip':
        from modeling_siglip import DualSigLIPVisionEncoder
        encoder = DualSigLIPVisionEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Load pretrained weights if provided
    if pretrained_path:
        encoder.load_pretrained_weights(pretrained_path)
    
    return encoder


def _create_config_from_pretrained(
    encoder_type: str,
    pretrained_path: str,
    **kwargs
):
    """Create DualVisionConfig from pretrained model config.
    
    Args:
        encoder_type: Type of encoder ('vit', 'clip', 'siglip')
        pretrained_path: Path to pretrained model
        **kwargs: Additional arguments to override
        
    Returns:
        DualVisionConfig with parameters from pretrained model
    """
    from config import DualVisionConfig
    
    try:
        if encoder_type == 'vit':
            from transformers import ViTConfig
            pretrained_config = ViTConfig.from_pretrained(pretrained_path)
            
            config_dict = {
                'encoder_type': 'vit',
                'hidden_size': pretrained_config.hidden_size,
                'num_layers': pretrained_config.num_hidden_layers,
                'num_heads': pretrained_config.num_attention_heads,
                'intermediate_size': pretrained_config.intermediate_size,
                'image_size': pretrained_config.image_size,
                'patch_size': pretrained_config.patch_size,
                'num_channels': pretrained_config.num_channels,
                'layer_norm_eps': pretrained_config.layer_norm_eps,
                'use_cls_token': True,
                'pooling_type': 'cls',
            }
            
        elif encoder_type == 'clip':
            from transformers import CLIPVisionConfig
            clip_config = CLIPVisionConfig.from_pretrained(pretrained_path)
            
            config_dict = {
                'encoder_type': 'clip',
                'hidden_size': clip_config.hidden_size,
                'num_layers': clip_config.num_hidden_layers,
                'num_heads': clip_config.num_attention_heads,
                'intermediate_size': clip_config.intermediate_size,
                'image_size': clip_config.image_size,
                'patch_size': clip_config.patch_size,
                'num_channels': clip_config.num_channels,
                'layer_norm_eps': clip_config.layer_norm_eps,
                'use_cls_token': True,
                'pooling_type': 'cls',
            }
            
        elif encoder_type == 'siglip':
            from transformers import SiglipVisionConfig
            siglip_config = SiglipVisionConfig.from_pretrained(pretrained_path)
            
            config_dict = {
                'encoder_type': 'siglip',
                'hidden_size': siglip_config.hidden_size,
                'num_layers': siglip_config.num_hidden_layers,
                'num_heads': siglip_config.num_attention_heads,
                'intermediate_size': siglip_config.intermediate_size,
                'image_size': siglip_config.image_size,
                'patch_size': siglip_config.patch_size,
                'num_channels': siglip_config.num_channels,
                'layer_norm_eps': siglip_config.layer_norm_eps,
                'use_cls_token': False,
                'pooling_type': 'mean',
                'use_bias_in_attention': False,
            }
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Override with user-provided kwargs
        config_dict.update(kwargs)
        
        return DualVisionConfig(**config_dict)
        
    except Exception as e:
        warnings.warn(
            f"Could not load config from pretrained model: {e}. "
            f"Falling back to default config."
        )
        return DualVisionConfig(encoder_type=encoder_type, **kwargs)
