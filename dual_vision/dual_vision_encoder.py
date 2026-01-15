"""Base Dual Vision Encoder implementation"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import warnings

try:
    from .config import DualVisionConfig
    from .attention import DualBranchAttention, DualBranchMLP
    from .utils import pad_auxiliary_tokens, generate_random_tokens
except ImportError:
    from config import DualVisionConfig
    from attention import DualBranchAttention, DualBranchMLP
    from utils import pad_auxiliary_tokens, generate_random_tokens


class DualVisionEncoderLayer(nn.Module):
    """Transformer layer with dual branches for image and auxiliary tokens.
    
    Architecture:
        Left branch:  LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
        Right branch: LayerNorm_mot -> Attention -> Residual -> LayerNorm_mot -> MLP_mot -> Residual
        
    Both branches share joint attention but have separate parameters.
    """
    
    def __init__(self, config: DualVisionConfig):
        super().__init__()
        self.config = config
        
        # Attention (joint for both branches)
        self.attention = DualBranchAttention(config)
        
        # Layer norms for left branch
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Layer norms for right branch (with _mot suffix)
        self.input_layernorm_mot = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm_mot = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP for left branch
        self.mlp = DualBranchMLP(config, is_mot=False)
        
        # MLP for right branch (with _mot suffix)
        self.mlp_mot = DualBranchMLP(config, is_mot=True)
    
    def forward(
        self,
        hidden_states_left: torch.Tensor,
        hidden_states_right: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states_left: [batch_size, seq_len_left, hidden_size]
            hidden_states_right: [batch_size, seq_len_right, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (left_output, right_output)
        """
        # Pre-norm for both branches
        normed_left = self.input_layernorm(hidden_states_left)
        normed_right = self.input_layernorm_mot(hidden_states_right)
        
        # Joint attention
        attn_output_left, attn_output_right = self.attention(
            normed_left, normed_right, attention_mask
        )
        
        # Residual connection
        hidden_states_left = hidden_states_left + attn_output_left
        hidden_states_right = hidden_states_right + attn_output_right
        
        # MLP with pre-norm
        normed_left = self.post_attention_layernorm(hidden_states_left)
        normed_right = self.post_attention_layernorm_mot(hidden_states_right)
        
        mlp_output_left = self.mlp(normed_left)
        mlp_output_right = self.mlp_mot(normed_right)
        
        # Residual connection
        hidden_states_left = hidden_states_left + mlp_output_left
        hidden_states_right = hidden_states_right + mlp_output_right
        
        return hidden_states_left, hidden_states_right


class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings."""
    
    def __init__(self, config: DualVisionConfig):
        super().__init__()
        self.config = config
        self.num_patches = config.num_patches
        
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch_size, num_channels, height, width]
            
        Returns:
            Patch embeddings [batch_size, num_patches, hidden_size]
        """
        batch_size = pixel_values.shape[0]
        embeddings = self.projection(pixel_values)  # [B, hidden_size, H/P, W/P]
        embeddings = embeddings.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        return embeddings


class DualVisionEncoder(nn.Module):
    """Base Dual Vision Encoder with MoT architecture.
    
    Can be subclassed for specific vision encoders (ViT, CLIP, SigLIP).
    """
    
    def __init__(self, config: DualVisionConfig):
        super().__init__()
        self.config = config
        self._preprocessor = None  # Will be set when loading pretrained weights
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # CLS token (if used)
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.cls_token_mot = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Positional embeddings
        if config.use_absolute_position_embeddings:
            # Left branch: 2D positional embedding (same as base vision encoder)
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, config.num_positions, config.hidden_size)
            )
            # Right branch: 1D learnable positional embedding (like CLIP text branch)
            # This is NOT a copy of left branch, but learned independently
            self.position_embedding_mot = nn.Embedding(
                config.num_positions, config.hidden_size
            )
        
        # Learnable auxiliary tokens for right branch (default input when no auxiliary_tokens provided)
        if config.use_learnable_tokens:
            self.learnable_auxiliary_tokens = nn.Parameter(
                torch.zeros(1, config.num_patches, config.hidden_size)
            )
            # Initialize with normal distribution (like other learnable embeddings)
            nn.init.normal_(self.learnable_auxiliary_tokens, mean=0.0, std=config.random_token_std)
        
        # Pre-layer norms (applied to embeddings before transformer layers)
        # CLIP applies layer norm to combined patch+position embeddings before encoder
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layrnorm_mot = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DualVisionEncoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norms (applied to pooled output)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layernorm_mot = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)
    
    def freeze_left_branch(self):
        """Freeze all left branch (image) parameters."""
        for name, param in self.named_parameters():
            if '_mot' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_left_parameters(self):
        """Get all left branch parameters (without _mot suffix)."""
        return [p for n, p in self.named_parameters() if '_mot' not in n and p.requires_grad]
    
    def get_right_parameters(self):
        """Get all right branch parameters (with _mot suffix)."""
        return [p for n, p in self.named_parameters() if '_mot' in n and p.requires_grad]
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        auxiliary_tokens: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, dict]:
        """
        Forward pass through dual vision encoder.
        
        Args:
            pixel_values: Input images [batch_size, num_channels, height, width]
            auxiliary_tokens: Optional auxiliary tokens [batch_size, seq_len, hidden_size]
                             If None, random tokens will be generated
            return_dict: Whether to return dict or tuple
            
        Returns:
            If return_dict=True:
                {
                    'left_branch_output': Left branch features,
                    'right_branch_output': Right branch features,
                    'pooled_output': Pooled output (based on config.output_mode)
                }
            If return_dict=False:
                Tuple of (pooled_output, left_output, right_output)
        """
        batch_size = pixel_values.shape[0]
        
        # ===== Left branch (image tokens) =====
        # Get patch embeddings
        left_embeddings = self.patch_embedding(pixel_values)  # [B, num_patches, hidden_size]
        
        # ===== Right branch (auxiliary tokens) =====
        # Generate or use provided auxiliary tokens
        if auxiliary_tokens is None:
            # Use learnable tokens as default (replaces random tokens)
            if self.config.use_learnable_tokens and hasattr(self, 'learnable_auxiliary_tokens'):
                # Expand learnable tokens to batch size
                auxiliary_tokens = self.learnable_auxiliary_tokens.expand(batch_size, -1, -1)
                auxiliary_tokens = auxiliary_tokens.to(device=pixel_values.device, dtype=pixel_values.dtype)
            else:
                # Fallback to random tokens
                seq_len = left_embeddings.shape[1]
                auxiliary_tokens = generate_random_tokens(
                    batch_size=batch_size,
                    seq_length=seq_len,
                    hidden_dim=self.config.hidden_size,
                    distribution=self.config.auxiliary_token_init,
                    std=self.config.random_token_std,
                    device=pixel_values.device,
                    dtype=pixel_values.dtype
                )
        else:
            # Pad or truncate to match left branch length
            target_len = left_embeddings.shape[1]
            if auxiliary_tokens.shape[1] != target_len:
                auxiliary_tokens = pad_auxiliary_tokens(auxiliary_tokens, target_len)
        right_embeddings = auxiliary_tokens
        
        # Add CLS token if used
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            left_embeddings = torch.cat([cls_tokens, left_embeddings], dim=1)
            cls_tokens = self.cls_token_mot.expand(batch_size, -1, -1)
            right_embeddings = torch.cat([cls_tokens, right_embeddings], dim=1)
        
        # Add positional embeddings
        if self.config.use_absolute_position_embeddings:
            left_embeddings = left_embeddings + self.position_embeddings
        
        left_embeddings = self.dropout(left_embeddings)
        
        # Add positional embeddings for right branch (1D learnable)
        if self.config.use_absolute_position_embeddings:
            # Create position ids for right branch
            seq_len = right_embeddings.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=right_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            # Apply 1D positional embedding
            right_embeddings = right_embeddings + self.position_embedding_mot(position_ids)
        
        right_embeddings = self.dropout(right_embeddings)
        
        # Apply pre-layer normalization (CLIP adds this after embeddings)
        left_embeddings = self.pre_layrnorm(left_embeddings)
        right_embeddings = self.pre_layrnorm_mot(right_embeddings)
        
        # ===== Pass through transformer layers =====
        hidden_states_left = left_embeddings
        hidden_states_right = right_embeddings
        
        for layer in self.layers:
            hidden_states_left, hidden_states_right = layer(
                hidden_states_left, hidden_states_right
            )
        
        # ===== Pooling =====
        # Pool BEFORE final layer norm (matching CLIP's implementation)
        pooled_left = self._pool_features(hidden_states_left)
        pooled_right = self._pool_features(hidden_states_right)
        
        # Apply final layer norm to POOLED outputs (not all tokens)
        pooled_left = self.final_layernorm(pooled_left)
        pooled_right = self.final_layernorm_mot(pooled_right)
        
        # Select output based on config
        if self.config.output_mode == 'right':
            pooled_output = pooled_right
        elif self.config.output_mode == 'left':
            pooled_output = pooled_left
        elif self.config.output_mode == 'both':
            pooled_output = (pooled_left, pooled_right)
        elif self.config.output_mode == 'concat':
            pooled_output = torch.cat([pooled_left, pooled_right], dim=-1)
        else:
            raise ValueError(f"Unknown output_mode: {self.config.output_mode}")
        
        if return_dict:
            return {
                'pooled_output': pooled_output,
                'left_last_hidden_state': hidden_states_left,
                'right_last_hidden_state': hidden_states_right,
            }
        else:
            return pooled_output, hidden_states_left, hidden_states_right
    
    def _pool_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool features based on config."""
        if self.config.pooling_type == 'cls':
            # Use CLS token
            return hidden_states[:, 0]
        elif self.config.pooling_type == 'mean':
            # Mean pooling
            return hidden_states.mean(dim=1)
        elif self.config.pooling_type == 'none':
            # Return all tokens
            return hidden_states
        else:
            raise ValueError(f"Unknown pooling_type: {self.config.pooling_type}")
    
    @property
    def preprocessor(self):
        """Get the image preprocessor for this encoder.
        
        Returns the preprocessor that matches the pretrained model.
        Should be set during load_pretrained_weights().
        """
        if self._preprocessor is None:
            # Return default transforms if preprocessor not set
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return self._preprocessor
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights from a vision encoder.
        
        This is a base implementation that should be overridden by subclasses
        for encoder-specific weight loading logic.
        """
        raise NotImplementedError("Subclasses should implement load_pretrained_weights")
