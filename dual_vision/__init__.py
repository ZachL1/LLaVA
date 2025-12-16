"""
Dual Vision Encoder with Mixture-of-Tokens (MoT) Architecture

A dual-path vision encoder where:
- Left branch processes image patch tokens
- Right branch processes auxiliary tokens (text, LLM hidden states, or random noise)
- Both branches perform joint attention
- Supports initialization from ViT, CLIP, and SigLIP encoders
"""

from .config import DualVisionConfig
from .dual_vision_encoder import DualVisionEncoder, DualVisionEncoderLayer
from .modeling_vit import DualViTEncoder
from .modeling_clip import DualCLIPVisionEncoder
from .modeling_siglip import DualSigLIPVisionEncoder
from .utils import create_dual_encoder, load_vision_encoder_weights

__all__ = [
    'DualVisionConfig',
    'DualVisionEncoder',
    'DualVisionEncoderLayer',
    'DualViTEncoder',
    'DualCLIPVisionEncoder',
    'DualSigLIPVisionEncoder',
    'create_dual_encoder',
    'load_vision_encoder_weights',
]
