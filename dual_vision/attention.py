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


class DualBranchAttention(nn.Module):
    """Dual-branch attention with separate Q/K/V projections for each branch.
    
    Based on Bagel MoT implementation where both branches perform joint attention
    over concatenated tokens.
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
        Forward pass with joint attention over both branches.
        
        Args:
            hidden_states_left: Left branch tokens [batch_size, seq_len_left, hidden_size]
            hidden_states_right: Right branch tokens [batch_size, seq_len_right, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len_total, seq_len_total]
            
        Returns:
            Tuple of (left_output, right_output) with same shapes as inputs
        """
        batch_size, seq_len_left, _ = hidden_states_left.shape
        seq_len_right = hidden_states_right.shape[1]
        
        # Project left branch
        q_left = self.q_proj(hidden_states_left)  # [B, L_left, num_heads * head_dim]
        k_left = self.k_proj(hidden_states_left)
        v_left = self.v_proj(hidden_states_left)
        
        # Project right branch
        q_right = self.q_proj_mot(hidden_states_right)  # [B, L_right, num_heads * head_dim]
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
        q_joint = torch.cat([q_left, q_right], dim=1)  # [B, L_left + L_right, num_heads, head_dim]
        k_joint = torch.cat([k_left, k_right], dim=1)
        v_joint = torch.cat([v_left, v_right], dim=1)
        
        # Compute attention
        # Flash attention requires CUDA
        use_flash = self.use_flash_attention and q_joint.is_cuda
        
        if use_flash:
            # Flash attention expects [batch_size, seq_len, num_heads, head_dim]
            # FlashAttention only supports fp16 and bf16 data types
            original_dtype = q_joint.dtype
            if original_dtype not in [torch.float16, torch.bfloat16]:
                # Convert to fp16 (or bf16 if available) for flash attention
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                q_joint = q_joint.to(compute_dtype)
                k_joint = k_joint.to(compute_dtype)
                v_joint = v_joint.to(compute_dtype)
            
            attn_output = flash_attn_func(
                q_joint, k_joint, v_joint,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0,
                causal=False
            )
            
            # Convert back to original dtype if needed
            if original_dtype not in [torch.float16, torch.bfloat16]:
                attn_output = attn_output.to(original_dtype)
        else:
            # Standard scaled dot-product attention
            # Transpose to [batch_size, num_heads, seq_len, head_dim]
            q_joint = q_joint.transpose(1, 2)
            k_joint = k_joint.transpose(1, 2)
            v_joint = v_joint.transpose(1, 2)
            
            # Compute attention scores
            attn_scores = torch.matmul(q_joint, k_joint.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            # Softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v_joint)
            
            # Transpose back to [batch_size, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2)
        
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
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
