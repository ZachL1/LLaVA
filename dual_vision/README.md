# Dual Vision Encoder with MoT Architecture

A PyTorch implementation of a dual-path vision encoder using Mixture-of-Tokens (MoT) architecture. This encoder processes images through two parallel branches with joint attention, supporting initialization from popular vision encoders (ViT, CLIP, SigLIP).

## Features

- **Dual-branch architecture**: Left branch for image patches, right branch for auxiliary tokens
- **Joint attention mechanism**: Both branches attend to each other via concatenated attention
- **MoT parameter naming**: Right branch parameters use `_mot` suffix for easy identification
- **Flexible auxiliary input**: Random tokens, text embeddings, or LLM hidden states
- **Multiple encoder support**: ViT, CLIP Vision Encoder, and SigLIP
- **Pretrained weight loading**: Easy initialization from HuggingFace models
- **Adaptation training**: Train right branch to align with original encoder output

## Architecture

```
Input Image → Patch Embedding
                    ↓
            ┌───────┴───────┐
            ↓               ↓
    Left Branch        Right Branch
   (Image Tokens)    (Auxiliary Tokens)
            ↓               ↓
       LayerNorm       LayerNorm_mot
            ↓               ↓
       ├─── Joint Attention ───┤
       │    (Concatenated)     │
       ↓                       ↓
     O_proj              O_proj_mot
       ↓                       ↓
   Residual              Residual
       ↓                       ↓
       MLP                 MLP_mot
       ↓                       ↓
  Left Output          Right Output
                           ↓
                    (Default Output)
```

## Installation

```bash
cd /test/annan/LLaVA/dual_vision
pip install torch torchvision transformers timm
# Optional: for flash attention
pip install flash-attn --no-build-isolation
```

## Quick Start

### Basic Usage

```python
from dual_vision import create_dual_encoder

# Create CLIP-based dual encoder
encoder = create_dual_encoder(
    encoder_type='clip',
    pretrained_path='openai/clip-vit-base-patch32',
    output_mode='right'  # Use right branch output
)

# Forward pass with random auxiliary tokens
import torch
images = torch.randn(2, 3, 224, 224)
output = encoder(images)  # Returns right branch features

# Or provide custom auxiliary tokens
auxiliary_tokens = torch.randn(2, 50, 512)  # e.g., from text encoder
output = encoder(images, auxiliary_tokens=auxiliary_tokens)
```

### Training Adaptation

Train the right branch to align with the original encoder:

```bash
python train_adaptation.py \
  --encoder_type clip \
  --pretrained_model openai/clip-vit-base-patch32 \
  --data_path /path/to/images \
  --output_dir ./checkpoints \
  --num_epochs 10 \
  --batch_size 32 \
  --lr 1e-4
```

This will:
- Freeze left branch (image) parameters
- Train only right branch (`_mot`) parameters  
- Minimize difference between right branch output and teacher encoder output
- Save checkpoints to `./checkpoints/`

## Supported Encoders

### ViT (Vision Transformer)

```python
from dual_vision import DualViTEncoder, DualVisionConfig

config = DualVisionConfig(
    encoder_type='vit',
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    image_size=224,
    patch_size=16
)

encoder = DualViTEncoder(config)
encoder.load_pretrained_weights('google/vit-base-patch16-224')
```

### CLIP Vision Encoder

```python
from dual_vision import DualCLIPVisionEncoder

# Convenience method
encoder = DualCLIPVisionEncoder.from_pretrained(
    'openai/clip-vit-base-patch32',
    output_mode='right'
)
```

### SigLIP Vision Encoder

```python
from dual_vision import DualSigLIPVisionEncoder

encoder = DualSigLIPVisionEncoder.from_pretrained(
    'google/siglip-base-patch16-224',
    output_mode='right'
)
```

## Configuration

Key configuration options:

```python
config = DualVisionConfig(
    encoder_type='vit',              # 'vit', 'clip', or 'siglip'
    hidden_size=768,                 # Hidden dimension
    num_layers=12,                   # Number of transformer layers
    num_heads=12,                    # Number of attention heads
    image_size=224,                  # Input image size
    patch_size=16,                   # Patch size
    
    # Auxiliary token settings
    auxiliary_token_init='gaussian', # 'gaussian' or 'uniform'
    random_token_std=0.02,           # Std for random tokens
    
    # Output settings
    output_mode='right',             # 'right', 'left', 'both', or 'concat'
    pooling_type='cls',              # 'cls', 'mean', or 'none'
    
    # Training settings
    freeze_left_branch=False,        # Freeze left branch
    use_flash_attention=True,        # Use flash attention
)
```

## Output Modes

- `'right'`: Return right branch features only (default)
- `'left'`: Return left branch features only
- `'both'`: Return tuple of (left, right) features
- `'concat'`: Return concatenated features

## Parameter Management

```python
# Freeze left branch for training right branch only
encoder.freeze_left_branch()

# Get parameters for different branches
left_params = encoder.get_left_parameters()
right_params = encoder.get_right_parameters()

# Create optimizer with different learning rates
from utils import get_parameter_groups
param_groups = get_parameter_groups(
    encoder,
    left_branch_lr=1e-5,
    right_branch_lr=1e-4
)
optimizer = torch.optim.AdamW(param_groups)
```

## Examples

See the `examples/` directory for complete usage examples:

- `test_vit_encoder.py`: ViT encoder examples
- `test_clip_encoder.py`: CLIP encoder with text features
- `test_siglip_encoder.py`: SigLIP encoder examples

Run tests:

```bash
python examples/test_vit_encoder.py
python examples/test_clip_encoder.py
python examples/test_siglip_encoder.py
```

## File Structure

```
dual_vision/
├── __init__.py                 # Package exports
├── config.py                   # Configuration classes
├── utils.py                    # Utility functions
├── attention.py                # MoT attention implementation
├── dual_vision_encoder.py      # Base encoder class
├── modeling_vit.py             # ViT-specific implementation
├── modeling_clip.py            # CLIP-specific implementation
├── modeling_siglip.py          # SigLIP-specific implementation
├── train_adaptation.py         # Adaptation training script
└── examples/                   # Usage examples
    ├── test_vit_encoder.py
    ├── test_clip_encoder.py
    └── test_siglip_encoder.py
```

## Key Design Decisions

1. **MoT Parameter Naming**: Right branch parameters use `_mot` suffix (e.g., `q_proj_mot`, `mlp_mot`) for easy identification and selective training

2. **Joint Attention**: Both branches are concatenated before attention computation, allowing full interaction between image and auxiliary tokens

3. **Random Token Generation**: When no auxiliary tokens provided, random tokens are generated on-the-fly during forward pass

4. **Flexible Initialization**: Supports loading from HuggingFace transformers, timm, or custom checkpoints

## Citation

This implementation is based on:
- Bagel MoT architecture: https://github.com/ByteDance-Seed/Bagel
- Vision Transformers (ViT)
- CLIP Vision Encoder
- SigLIP Vision Encoder

## License

This code follows the same license as the original vision encoder implementations it's based on.
