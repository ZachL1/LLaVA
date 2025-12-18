"""Configuration classes for Dual Vision Encoder"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DualVisionConfig:
    """Configuration for Dual Vision Encoder with MoT architecture.
    
    Args:
        encoder_type: Type of vision encoder ('vit', 'clip', 'siglip')
        hidden_size: Dimension of hidden representations
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of MLP intermediate layer
        image_size: Input image size (height, width)
        patch_size: Size of image patches
        num_channels: Number of input image channels
        
        # MoT-specific settings
        use_qk_norm: Whether to use QK normalization in attention
        auxiliary_token_init: How to initialize auxiliary tokens ('random', 'gaussian', 'uniform')
        random_token_std: Standard deviation for Gaussian random tokens
        
        # Output settings
        output_mode: Which branch output to return ('right', 'left', 'both', 'concat')
        pooling_type: Pooling strategy ('cls', 'mean', 'none')
        
        # Training settings
        freeze_left_branch: Whether to freeze left branch parameters
        use_flash_attention: Whether to use flash attention for efficiency
        
        # Architecture details
        layer_norm_eps: Epsilon for layer normalization
        dropout_prob: Dropout probability
        attention_dropout_prob: Attention dropout probability
        use_bias_in_attention: Whether to use bias in attention projections
        use_bias_in_mlp: Whether to use bias in MLP layers
        
        # Positional encoding
        use_absolute_position_embeddings: Whether to use absolute positional embeddings
        use_cls_token: Whether to use CLS token (for ViT/CLIP style encoders)
    """
    
    # Basic architecture
    encoder_type: Literal['vit', 'clip', 'siglip'] = 'vit'
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: Optional[int] = None
    
    # Image settings
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    
    # MoT-specific
    use_qk_norm: bool = False
    auxiliary_token_init: Literal['random', 'gaussian', 'uniform'] = 'gaussian'
    random_token_std: float = 0.02
    attention_mode: Literal['joint', 'cross'] = 'joint'  # joint: both branches attend to each other; cross: right queries left
    
    # Output settings
    output_mode: Literal['right', 'left', 'both', 'concat'] = 'right'
    pooling_type: Literal['cls', 'mean', 'none'] = 'cls'
    
    # Training settings
    freeze_left_branch: bool = False
    use_flash_attention: bool = True
    
    # Architecture details
    layer_norm_eps: float = 1e-6
    dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    use_bias_in_attention: bool = True
    use_bias_in_mlp: bool = True
    
    # Positional encoding
    use_absolute_position_embeddings: bool = True
    use_cls_token: bool = True
    
    def __post_init__(self):
        """Set default values based on encoder type"""
        # Set intermediate_size if not specified
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        
        # Adjust settings based on encoder type
        if self.encoder_type == 'clip':
            # CLIP uses pre-LayerNorm and CLS token
            self.use_cls_token = True
        elif self.encoder_type == 'siglip':
            # SigLIP doesn't use CLS token, uses mean pooling
            self.use_cls_token = False
            self.pooling_type = 'mean'
            self.use_bias_in_attention = False
    
    @property
    def num_patches(self) -> int:
        """Number of patches in the image"""
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def num_positions(self) -> int:
        """Total number of positions (patches + CLS token if used)"""
        return self.num_patches + (1 if self.use_cls_token else 0)
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head"""
        return self.hidden_size // self.num_heads
