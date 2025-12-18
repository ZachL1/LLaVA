"""MoT Attention layer for Dual Vision Encoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class QuickGELU(nn.Module):
    """Quick GELU activation used by CLIP.
    
    Formula: x * sigmoid(1.702 * x)
    This is an approximation of GELU that's faster to compute.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)



class DualBranchAttention(nn.Module):
    """Dual-branch attention with separate Q/K/V projections for each branch.
    
    Supports two attention modes:
    1. Joint attention: Both branches attend to concatenated tokens (default)
    2. cross attention: Right branch (text) queries left branch (vision) for cross-attention
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTN_AVAILABLE
        
        # Left branch projections (image tokens)
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.use_bias_in_attention
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias_in_attention
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias_in_attention
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias_in_attention
        )
        
        # Right branch projections (auxiliary tokens) - with _mot suffix
        self.q_proj_mot = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias_in_attention
        )
        self.k_proj_mot = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias_in_attention
        )
        self.v_proj_mot = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias_in_attention
        )
        self.o_proj_mot = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias_in_attention
        )
        
        # QK normalization (optional)
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
            self.q_norm_mot = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm_mot = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_mot = nn.Identity()
            self.k_norm_mot = nn.Identity()
        
        # Attention dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)
    
    def forward(
        self,
        hidden_states_left: torch.Tensor,
        hidden_states_right: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-branch attention.
        
        Supports two modes:
        1. Joint attention (default): Both branches attend to concatenated tokens
        2. cross attention: Right branch (text) queries left branch (vision) for feature selection
        
        Args:
            hidden_states_left: Left branch tokens [batch_size, seq_len_left, hidden_size]
            hidden_states_right: Right branch tokens [batch_size, seq_len_right, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (left_output, right_output) with same shapes as inputs
        """
        batch_size, seq_len_left, _ = hidden_states_left.shape
        seq_len_right = hidden_states_right.shape[1]
        
        if self.config.attention_mode == 'joint':
            # Original joint attention: both branches attend to each other
            return self._joint_attention(
                hidden_states_left, hidden_states_right, 
                batch_size, seq_len_left, seq_len_right, 
                attention_mask
            )
        elif self.config.attention_mode == 'cross':
            # cross attention: right branch queries left branch
            return self._cross_attention(
                hidden_states_left, hidden_states_right,
                batch_size, seq_len_left, seq_len_right,
                attention_mask
            )
        else:
            raise ValueError(f"Unknown attention_mode: {self.config.attention_mode}")
    
    def _joint_attention(
        self,
        hidden_states_left: torch.Tensor,
        hidden_states_right: torch.Tensor,
        batch_size: int,
        seq_len_left: int,
        seq_len_right: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint attention: both branches attend to concatenated tokens."""
        # Project left branch
        q_left = self.q_proj(hidden_states_left)
        k_left = self.k_proj(hidden_states_left)
        v_left = self.v_proj(hidden_states_left)
        
        # Project right branch
        q_right = self.q_proj_mot(hidden_states_right)
        k_right = self.k_proj_mot(hidden_states_right)
        v_right = self.v_proj_mot(hidden_states_right)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q_left = q_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        k_left = k_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        v_left = v_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        
        q_right = q_right.view(batch_size, seq_len_right, self.num_heads, self.head_dim)
        k_right = k_right.view(batch_size, seq_len_right, self.num_heads, self.head_dim)
        v_right = v_right.view(batch_size, seq_len_right, self.num_heads, self.head_dim)
        
        # Apply QK normalization if enabled
        q_left = self.q_norm(q_left)
        k_left = self.k_norm(k_left)
        q_right = self.q_norm_mot(q_right)
        k_right = self.k_norm_mot(k_right)
        
        # Concatenate for joint attention
        q_joint = torch.cat([q_left, q_right], dim=1)
        k_joint = torch.cat([k_left, k_right], dim=1)
        v_joint = torch.cat([v_left, v_right], dim=1)
        
        # Compute attention
        attn_output = self._compute_attention(q_joint, k_joint, v_joint, attention_mask)
        
        # Split outputs back to left and right branches
        attn_output_left = attn_output[:, :seq_len_left, :, :]
        attn_output_right = attn_output[:, seq_len_left:, :, :]
        
        # Reshape and project
        attn_output_left = attn_output_left.contiguous().view(
            batch_size, seq_len_left, self.num_heads * self.head_dim
        )
        attn_output_right = attn_output_right.contiguous().view(
            batch_size, seq_len_right, self.num_heads * self.head_dim
        )
        
        # Separate output projections for each branch
        output_left = self.o_proj(attn_output_left)
        output_right = self.o_proj_mot(attn_output_right)
        
        return output_left, output_right
    
    def _cross_attention(
        self,
        hidden_states_left: torch.Tensor,
        hidden_states_right: torch.Tensor,
        batch_size: int,
        seq_len_left: int,
        seq_len_right: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cross attention: Right branch (text) queries left branch (vision).
        
        This implements cross-attention where:
        - Left branch: Self-attention using q_proj, k_proj, v_proj, o_proj
        - Right branch: Cross-attention where
          * Q comes from right branch (text features) using q_proj_mot
          * K, V come from left branch (vision features) using k_proj, v_proj
          * Output projection using o_proj_mot
        
        This allows the text features to selectively query and extract relevant vision features.
        """
        # Left branch: Self-attention
        q_left = self.q_proj(hidden_states_left)
        k_left = self.k_proj(hidden_states_left)
        v_left = self.v_proj(hidden_states_left)
        
        q_left = q_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        k_left = k_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        v_left = v_left.view(batch_size, seq_len_left, self.num_heads, self.head_dim)
        
        q_left = self.q_norm(q_left)
        k_left = self.k_norm(k_left)
        
        # Left branch self-attention
        attn_output_left = self._compute_attention(q_left, k_left, v_left, attention_mask)
        
        attn_output_left = attn_output_left.contiguous().view(
            batch_size, seq_len_left, self.num_heads * self.head_dim
        )
        output_left = self.o_proj(attn_output_left)
        
        # Right branch: Cross-attention to left branch
        # Q from right branch (text)
        q_right = self.q_proj_mot(hidden_states_right)
        q_right = q_right.view(batch_size, seq_len_right, self.num_heads, self.head_dim)
        q_right = self.q_norm_mot(q_right)
        
        # K, V from left branch (vision) - reuse the already computed k_left, v_left
        # This enables text to query vision features
        
        # Cross-attention: text queries vision
        attn_output_right = self._compute_attention(q_right, k_left, v_left, attention_mask)
        
        attn_output_right = attn_output_right.contiguous().view(
            batch_size, seq_len_right, self.num_heads * self.head_dim
        )
        output_right = self.o_proj_mot(attn_output_right)
        
        return output_left, output_right
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention given Q, K, V tensors.
        
        Args:
            q: Query [batch_size, seq_len_q, num_heads, head_dim]
            k: Key [batch_size, seq_len_k, num_heads, head_dim]
            v: Value [batch_size, seq_len_k, num_heads, head_dim]
            attention_mask: Optional mask
            
        Returns:
            Attention output [batch_size, seq_len_q, num_heads, head_dim]
        """
        # Flash attention requires CUDA
        use_flash = self.use_flash_attention and q.is_cuda
        
        if use_flash:
            # Flash attention expects [batch_size, seq_len, num_heads, head_dim]
            # FlashAttention only supports fp16 and bf16 data types
            original_dtype = q.dtype
            if original_dtype not in [torch.float16, torch.bfloat16]:
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)
            
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0,
                causal=False
            )
            
            if original_dtype not in [torch.float16, torch.bfloat16]:
                attn_output = attn_output.to(original_dtype)
        else:
            # Standard scaled dot-product attention
            # Transpose to [batch_size, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)
            
            # Transpose back to [batch_size, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2)
        
        return attn_output


class DualBranchMLP(nn.Module):
    """MLP layer with separate parameters for each branch."""
    
    def __init__(self, config, is_mot: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        suffix = '_mot' if is_mot else ''
        
        self.fc1 = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.use_bias_in_mlp
        )
        self.fc2 = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.use_bias_in_mlp
        )
        
        # Use QuickGELU for CLIP, standard GELU for others
        if config.encoder_type == 'clip':
            # Assuming QuickGELU is imported or defined elsewhere in the full context
            # For example: from transformers.activations import QuickGELU
            self.activation = QuickGELU()
        else:
            self.activation = nn.GELU()
        
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
