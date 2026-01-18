"""
ControlNet-style Vision Tower for LLaVA.

Design:
- Left branch: Frozen pretrained vision encoder (CLIP/SigLIP)
- Right branch: Trainable text-conditioned processor (only receives text hidden states)
- Zero fusion: Cross-attention based fusion with zero-initialized output projection
- Output: Left branch output (gradually influenced by right branch during training)

Architecture:
    Input Image ──────────────────────────────────────► Left Branch (Frozen)
                                                              │
    Text Hidden State ──► llm_hidden_proj ──► Right Branch (Trainable)
                                              │               │
                                              │               │
    For each layer i:                         ▼               ▼
                                        right_layer_i   left_layer_i
                                              │               │
                                              ▼               │
                                        zero_fusion_i         │
                                        (cross-attn)          │
                                              │               │
                                              └──────► (+) ◄──┘
                                                       │
                                                       ▼
                                                 left_hidden
                                                       │
                                                       ▼
                                               Final Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from transformers import CLIPVisionConfig, CLIPImageProcessor


# ============================================================================
# Activation Functions
# ============================================================================

class QuickGELU(nn.Module):
    """Quick GELU activation used by CLIP."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# ============================================================================
# Basic Building Blocks
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert images to patch embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        embeddings = self.projection(pixel_values)  # [B, hidden_size, H/P, W/P]
        embeddings = embeddings.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        return embeddings


class TransformerMLP(nn.Module):
    """Standard transformer MLP block."""
    
    def __init__(self, config, use_quick_gelu: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = QuickGELU() if use_quick_gelu else nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """Standard self-attention layer."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [B, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class TransformerLayer(nn.Module):
    """Standard transformer layer with pre-norm."""
    
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TransformerMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# Zero Fusion Layer (ControlNet-style)
# ============================================================================

class ZeroCrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion with zero-initialized output projection.
    
    Left tokens (vision) query right tokens (text condition).
    Output is added to left branch.
    
    Initially outputs zeros due to zero initialization, ensuring no impact
    on left branch at the start of training.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Query from left branch (vision)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # Key and Value from right branch (text condition)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # Output projection - ZERO INITIALIZED
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Zero initialization for output projection
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self, 
        left_hidden: torch.Tensor,  # [B, seq_len_left, hidden_size]
        right_hidden: torch.Tensor  # [B, seq_len_right, hidden_size]
    ) -> torch.Tensor:
        """
        Cross-attention: left queries right.
        
        Returns:
            Tensor of shape [B, seq_len_left, hidden_size] to be added to left branch
        """
        batch_size, seq_len_left, _ = left_hidden.shape
        seq_len_right = right_hidden.shape[1]
        
        # Query from left, Key/Value from right
        q = self.q_proj(left_hidden)
        k = self.k_proj(right_hidden)
        v = self.v_proj(right_hidden)
        
        # Reshape to [B, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len_left, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_right, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_right, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project through zero-initialized layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_left, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


# ============================================================================
# ControlNet Vision Encoder
# ============================================================================

class ControlNetVisionConfig:
    """Configuration for ControlNet Vision Encoder."""
    
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        dropout_prob: float = 0.0,
        max_text_seq_len: int = 2048,  # Max text sequence length for right branch
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.dropout_prob = dropout_prob
        self.max_text_seq_len = max_text_seq_len
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
    
    @classmethod
    def from_clip_config(cls, clip_config: CLIPVisionConfig, **kwargs):
        """Create ControlNetVisionConfig from CLIPVisionConfig."""
        return cls(
            hidden_size=clip_config.hidden_size,
            intermediate_size=clip_config.intermediate_size,
            num_hidden_layers=clip_config.num_hidden_layers,
            num_attention_heads=clip_config.num_attention_heads,
            image_size=clip_config.image_size,
            patch_size=clip_config.patch_size,
            num_channels=clip_config.num_channels,
            layer_norm_eps=clip_config.layer_norm_eps,
            attention_dropout=getattr(clip_config, 'attention_dropout', 0.0),
            dropout_prob=getattr(clip_config, 'hidden_dropout_prob', 0.0),
            **kwargs
        )


class ControlNetVisionEncoder(nn.Module):
    """
    ControlNet-style Vision Encoder.
    
    Architecture:
    - Left branch: Frozen pretrained vision encoder
    - Right branch: Trainable text-conditioned processor
    - Zero fusion: Cross-attention fusion with zero-initialized output
    """
    
    def __init__(self, config: ControlNetVisionConfig):
        super().__init__()
        self.config = config
        
        # ==================== Left Branch (Frozen) ====================
        # Patch embedding for images
        self.patch_embedding = PatchEmbedding(config)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Position embeddings for image patches + CLS
        num_positions = config.num_patches + 1  # patches + CLS
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        
        # Pre-layer norm (CLIP-style)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Left transformer layers (frozen)
        self.left_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # ==================== Right Branch (Trainable) ====================
        # Position embeddings for text tokens (1D learnable)
        self.right_pos_embedding = nn.Embedding(config.max_text_seq_len, config.hidden_size)
        
        # Pre-layer norm for right branch
        self.right_pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Right transformer layers (trainable)
        self.right_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # ==================== Zero Fusion Layers ====================
        # Cross-attention: left queries right, zero-initialized output
        self.zero_fusions = nn.ModuleList([
            ZeroCrossAttentionFusion(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.right_pos_embedding.weight, std=0.02)
    
    def freeze_left_branch(self):
        """Freeze all left branch parameters."""
        # Freeze patch embedding
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        
        # Freeze CLS token and position embeddings
        self.cls_token.requires_grad = False
        self.position_embeddings.requires_grad = False
        
        # Freeze pre-layer norm
        for param in self.pre_layernorm.parameters():
            param.requires_grad = False
        
        # Freeze left transformer layers
        for layer in self.left_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze final layer norm
        for param in self.final_layernorm.parameters():
            param.requires_grad = False
        
        print("Left branch frozen successfully")
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters (right branch + zero fusions)."""
        trainable = []
        
        # Right branch
        trainable.extend(self.right_pos_embedding.parameters())
        trainable.extend(self.right_pre_layernorm.parameters())
        for layer in self.right_layers:
            trainable.extend(layer.parameters())
        
        # Zero fusions
        for fusion in self.zero_fusions:
            trainable.extend(fusion.parameters())
        
        return trainable
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_hidden_states: torch.Tensor,
        output_hidden_states: bool = False
    ) -> dict:
        """
        Forward pass.
        
        Args:
            pixel_values: Image tensor [B, C, H, W]
            text_hidden_states: Text condition from LLM [B, text_seq_len, hidden_size]
                                (already projected to vision hidden size)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            dict with:
                - last_hidden_state: Final output [B, num_patches+1, hidden_size]
                - pooler_output: CLS token output [B, hidden_size]
                - hidden_states: (optional) all layer outputs
        """
        batch_size = pixel_values.shape[0]
        
        # ==================== Left Branch: Process Image ====================
        # Patch embedding
        left_hidden = self.patch_embedding(pixel_values)  # [B, num_patches, hidden_size]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        left_hidden = torch.cat([cls_tokens, left_hidden], dim=1)  # [B, num_patches+1, hidden_size]
        
        # Add position embeddings
        left_hidden = left_hidden + self.position_embeddings
        
        # Pre-layer norm
        left_hidden = self.pre_layernorm(left_hidden)
        
        # ==================== Right Branch: Process Text Condition ====================
        text_seq_len = text_hidden_states.shape[1]
        
        # Add position embeddings
        position_ids = torch.arange(text_seq_len, device=text_hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        right_hidden = text_hidden_states + self.right_pos_embedding(position_ids)
        
        # Pre-layer norm
        right_hidden = self.right_pre_layernorm(right_hidden)
        
        # ==================== Process Through Layers with Fusion ====================
        all_hidden_states = [] if output_hidden_states else None
        
        for left_layer, right_layer, zero_fusion in zip(
            self.left_layers, self.right_layers, self.zero_fusions
        ):
            if output_hidden_states:
                all_hidden_states.append(left_hidden)
            
            # Left branch forward (frozen, but we still compute gradients for fusion)
            left_hidden = left_layer(left_hidden)
            
            # Right branch forward (trainable)
            right_hidden = right_layer(right_hidden)
            
            # Zero fusion: cross-attention from left to right
            fusion_output = zero_fusion(left_hidden, right_hidden)
            
            # Add fusion to left branch
            left_hidden = left_hidden + fusion_output
        
        # Final layer norm
        last_hidden_state = self.final_layernorm(left_hidden)
        
        if output_hidden_states:
            all_hidden_states.append(last_hidden_state)
        
        # Pooler output (CLS token)
        pooler_output = last_hidden_state[:, 0]
        
        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
            'hidden_states': tuple(all_hidden_states) if output_hidden_states else None
        }
    
    def load_pretrained_weights(self, pretrained_path: str, strict: bool = False):
        """
        Load pretrained CLIP vision encoder weights into left branch.
        Also copies weights to right branch as initialization.
        
        Args:
            pretrained_path: HuggingFace model name or path
            strict: Whether to strictly enforce key matching
        """
        from transformers import CLIPVisionModel
        
        print(f"Loading pretrained weights from: {pretrained_path}")
        
        # Load CLIP model
        clip_model = CLIPVisionModel.from_pretrained(pretrained_path)
        clip_state_dict = clip_model.state_dict()
        
        # Convert and load weights
        self._load_clip_weights(clip_state_dict)
        
        # Copy left branch weights to right branch
        self._copy_left_to_right()
        
        print("Pretrained weights loaded successfully")
    
    def _load_clip_weights(self, clip_state_dict: dict):
        """Load CLIP weights into left branch."""
        our_state_dict = {}
        
        for key, value in clip_state_dict.items():
            # Remove 'vision_model.' prefix
            if key.startswith('vision_model.'):
                key = key[13:]
            
            # Map embeddings
            if key == 'embeddings.class_embedding':
                our_state_dict['cls_token'] = value.unsqueeze(0).unsqueeze(0)
            elif key == 'embeddings.position_embedding.weight':
                our_state_dict['position_embeddings'] = value.unsqueeze(0)
            elif key.startswith('embeddings.patch_embedding'):
                new_key = key.replace('embeddings.patch_embedding', 'patch_embedding.projection')
                our_state_dict[new_key] = value
            
            # Map encoder layers to left_layers
            elif key.startswith('encoder.layers.'):
                parts = key.split('.')
                layer_idx = parts[2]
                rest = '.'.join(parts[3:])
                
                # Map attention
                if 'self_attn.q_proj' in rest:
                    new_key = f'left_layers.{layer_idx}.attention.q_proj' + rest.split('q_proj')[1]
                elif 'self_attn.k_proj' in rest:
                    new_key = f'left_layers.{layer_idx}.attention.k_proj' + rest.split('k_proj')[1]
                elif 'self_attn.v_proj' in rest:
                    new_key = f'left_layers.{layer_idx}.attention.v_proj' + rest.split('v_proj')[1]
                elif 'self_attn.out_proj' in rest:
                    new_key = f'left_layers.{layer_idx}.attention.o_proj' + rest.split('out_proj')[1]
                # Map layer norms
                elif 'layer_norm1' in rest:
                    new_key = f'left_layers.{layer_idx}.input_layernorm' + rest.split('layer_norm1')[1]
                elif 'layer_norm2' in rest:
                    new_key = f'left_layers.{layer_idx}.post_attention_layernorm' + rest.split('layer_norm2')[1]
                # Map MLP
                elif 'mlp.fc1' in rest:
                    new_key = f'left_layers.{layer_idx}.mlp.fc1' + rest.split('fc1')[1]
                elif 'mlp.fc2' in rest:
                    new_key = f'left_layers.{layer_idx}.mlp.fc2' + rest.split('fc2')[1]
                else:
                    continue
                
                our_state_dict[new_key] = value
            
            # Map pre/post layer norms
            elif 'pre_layrnorm' in key or 'pre_layernorm' in key:
                suffix = key.split('pre_layrnorm')[1] if 'pre_layrnorm' in key else key.split('pre_layernorm')[1]
                our_state_dict['pre_layernorm' + suffix] = value
            elif key.startswith('post_layernorm'):
                our_state_dict['final_layernorm' + key[14:]] = value
        
        # Load into model
        missing, unexpected = self.load_state_dict(our_state_dict, strict=False)
        
        print(f"Loaded {len(our_state_dict)} keys into left branch")
        if missing:
            # Filter out expected missing keys (right branch, zero fusions)
            unexpected_missing = [k for k in missing if not k.startswith(('right_', 'zero_'))]
            if unexpected_missing:
                print(f"Unexpected missing keys: {unexpected_missing[:10]}...")
    
    def _copy_left_to_right(self):
        """Copy left branch transformer layer weights to right branch."""
        print("Copying left branch weights to right branch...")
        
        for left_layer, right_layer in zip(self.left_layers, self.right_layers):
            # Copy state dict
            right_layer.load_state_dict(left_layer.state_dict())
        
        # Copy pre-layer norm
        self.right_pre_layernorm.load_state_dict(self.pre_layernorm.state_dict())
        
        print("Right branch initialized from left branch")


# ============================================================================
# ControlNet Vision Tower (LLaVA Integration)
# ============================================================================

class ControlNetVisionTower(nn.Module):
    """
    ControlNet-style Vision Tower wrapper for LLaVA integration.
    
    Features:
    - Left branch: Frozen pretrained vision encoder
    - Right branch: Trainable, receives text hidden states from LLM as condition
    - Zero fusion: Cross-attention based, zero-initialized for gradual influence
    - Output: Left branch output (enhanced by right branch during training)
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        
        # Parse vision_tower string
        # Format: "controlnet_vision_{encoder_type}_{model_path}"
        # Example: "controlnet_vision_clip_openai/clip-vit-large-patch14-336"
        if vision_tower.startswith("controlnet_vision_"):
            remainder = vision_tower[len("controlnet_vision_"):]
            parts = remainder.split('_', 1)
            if len(parts) >= 2:
                self.encoder_type = parts[0]
                self.pretrained_path = parts[1]
            else:
                self.encoder_type = 'clip'
                self.pretrained_path = getattr(args, 'controlnet_vision_pretrained', 'openai/clip-vit-large-patch14-336')
        else:
            self.encoder_type = 'clip'
            self.pretrained_path = getattr(args, 'controlnet_vision_pretrained', 'openai/clip-vit-large-patch14-336')
        
        # Get config args
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # LLM hidden state projection
        self.llm_hidden_proj = None
        self.llm_hidden_size = None
        
        # Create vision tower and image processor
        self._create_vision_tower_and_processor()
        
        # Initialize text projection if LLM hidden size is known
        llm_hidden_size = getattr(args, 'hidden_size', None)
        if llm_hidden_size is not None:
            self.set_llm_hidden_size(llm_hidden_size)
        
        # Load weights
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
    
    def _create_vision_tower_and_processor(self):
        """Create vision tower and image processor."""
        print(f"Creating ControlNet Vision Tower: encoder_type={self.encoder_type}, pretrained={self.pretrained_path}")
        
        if self.encoder_type == 'clip':
            # Load CLIP config
            clip_config = CLIPVisionConfig.from_pretrained(self.pretrained_path)
            
            # Create ControlNet config
            config = ControlNetVisionConfig.from_clip_config(clip_config)
            
            # Create encoder
            self.vision_tower = ControlNetVisionEncoder(config)
            
            # Get image processor
            self.image_processor = CLIPImageProcessor.from_pretrained(self.pretrained_path)
            
            # Store config
            self._config = config
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}. Currently only 'clip' is supported.")
        
        print(f"ControlNet Vision Tower structure created: {self.vision_tower_name}")
    
    def load_model(self, device_map=None):
        """Load pretrained weights."""
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return
        
        print(f"Loading weights for ControlNet Vision Tower: {self.vision_tower_name}")
        
        # Load pretrained weights
        self.vision_tower.load_pretrained_weights(self.pretrained_path)
        
        # Freeze left branch
        self.vision_tower.freeze_left_branch()
        
        self.is_loaded = True
        print(f"ControlNet Vision Tower loaded successfully: {self.vision_tower_name}")
    
    def set_llm_hidden_size(self, llm_hidden_size: int):
        """
        Initialize text projection layer.
        
        Args:
            llm_hidden_size: Hidden dimension of the LLM
        """
        self.llm_hidden_size = llm_hidden_size
        vision_hidden_size = self._config.hidden_size
        
        # Linear projection: LLM hidden dim -> Vision hidden dim
        self.llm_hidden_proj = nn.Linear(llm_hidden_size, vision_hidden_size)
        
        # Initialize with small weights
        nn.init.normal_(self.llm_hidden_proj.weight, std=0.02)
        nn.init.zeros_(self.llm_hidden_proj.bias)
        
        # Move to same device and dtype as vision tower
        self.llm_hidden_proj = self.llm_hidden_proj.to(
            device=self.device,
            dtype=self.dtype
        )
        
        print(f"Initialized text projection: {llm_hidden_size} -> {vision_hidden_size}")
    
    def _align_sequence_length(self, hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
        """Align text hidden states sequence length."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len == target_length:
            return hidden_states
        elif seq_len > target_length:
            # Truncate: keep last target_length tokens
            return hidden_states[:, -target_length:, :]
        else:
            # Pad: replicate last token
            padding_length = target_length - seq_len
            last_token = hidden_states[:, -1:, :].expand(-1, padding_length, -1)
            return torch.cat([hidden_states, last_token], dim=1)
    
    def forward(self, images, text_hidden_states=None):
        """
        Forward pass.
        
        Args:
            images: Image tensor [B, C, H, W] or list of tensors
            text_hidden_states: Optional LLM hidden states [B, seq_len, llm_hidden_dim]
                              If None, uses zero condition (no text influence)
        
        Returns:
            Image features [B, num_patches, hidden_size]
        """
        if isinstance(images, list):
            # Handle list of images
            image_features = []
            for idx, image in enumerate(images):
                # Get text hidden states for this image
                if text_hidden_states is not None and self.llm_hidden_proj is not None:
                    if isinstance(text_hidden_states, list):
                        text_hs = text_hidden_states[idx]
                    else:
                        text_hs = text_hidden_states[idx:idx+1]
                    
                    # Project to vision hidden dimension
                    text_cond = self.llm_hidden_proj(text_hs)
                    text_cond = text_cond.to(device=image.device, dtype=image.dtype)
                else:
                    # Use zero condition
                    text_cond = torch.zeros(
                        1, 1, self._config.hidden_size,
                        device=image.device, dtype=image.dtype
                    )
                
                # Forward through encoder
                output = self.vision_tower(
                    image.unsqueeze(0),
                    text_hidden_states=text_cond,
                    output_hidden_states=True
                )
                
                # Select features
                features = self._select_features(output)
                image_features.append(features)
            
            return image_features
        else:
            # Batch processing
            batch_size = images.shape[0]
            
            if text_hidden_states is not None and self.llm_hidden_proj is not None:
                # Project text hidden states
                text_cond = self.llm_hidden_proj(text_hidden_states)
                text_cond = text_cond.to(device=images.device, dtype=images.dtype)
            else:
                # Use zero condition
                text_cond = torch.zeros(
                    batch_size, 1, self._config.hidden_size,
                    device=images.device, dtype=images.dtype
                )
            
            # Forward through encoder
            output = self.vision_tower(
                images,
                text_hidden_states=text_cond,
                output_hidden_states=True
            )
            
            # Select features
            image_features = self._select_features(output)
            
            return image_features
    
    def _select_features(self, output: dict) -> torch.Tensor:
        """Select features based on select_layer and select_feature config."""
        if self.select_layer == -1:
            hidden_state = output['last_hidden_state']
        else:
            hidden_state = output['hidden_states'][self.select_layer]
        
        if self.select_feature == 'patch':
            # Remove CLS token
            return hidden_state[:, 1:]
        elif self.select_feature == 'cls_patch':
            return hidden_state
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
    
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
        return self._config.hidden_size
    
    @property
    def num_patches_per_side(self):
        return self._config.image_size // self._config.patch_size
    
    @property
    def num_patches(self):
        return self._config.num_patches


