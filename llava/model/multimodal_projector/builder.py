import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



class SwiGLUFFN(nn.Module):
    """
    A Feed-Forward Network using the Swish-Gated Linear Unit (SwiGLU).
    This has become the state-of-the-art FFN design in modern LLMs like LLaMA and PaLM.
    It uses a gating mechanism which provides more expressive power than a static activation.
    """
    def __init__(self, d_model: int, d_model_out: int, hidden_dim: int = None, dropout: float = 0.1):
        """
        Args:
            d_model (int): The input and output dimension of the model.
            hidden_dim (int): The intermediate dimension. To keep parameter counts similar to
                              a canonical FFN with 4*d_model expansion, this is often set to
                              approximately (2/3) * (4 * d_model).
            dropout (float): The dropout rate.
        """
        super().__init__()
        # Adjust hidden_dim if not provided, following the PaLM paper's heuristic
        if hidden_dim is None:
            hidden_dim = int(2/3 * 4 * d_model)

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model_out, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU FFN.
        SwiGLU-FFN(x) = W_down( (Swish(x * W_gate)) ⊙ (x * W_up) )
        Here, Swish is implemented by nn.functional.silu
        """
        gate = self.w_gate(x)
        up = self.w_up(x)
        
        # Element-wise multiplication of the gated and value paths
        fused_gate_up = F.silu(gate) * up
        
        fused_gate_up = self.dropout(fused_gate_up)
        
        # Final projection back to the model dimension
        output = self.w_down(fused_gate_up)
        
        return output

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    elif mlp_silu_match:
        return SwiGLUFFN(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
