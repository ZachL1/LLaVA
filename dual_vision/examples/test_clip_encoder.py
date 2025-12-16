"""
Example: Testing Dual CLIP Vision Encoder

This script demonstrates how to use the CLIP-based dual vision encoder.
"""

import torch
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import DualVisionConfig
from modeling_clip import DualCLIPVisionEncoder


def test_clip_encoder():
    """Test CLIP vision encoder."""
    print("=" * 60)
    print("Testing DualCLIPVisionEncoder")
    print("=" * 60)
    
    # Create config
    config = DualVisionConfig(
        encoder_type='clip',
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        image_size=224,
        patch_size=32,  # CLIP typically uses 32x32 patches
        output_mode='right',
    )
    
    # Create encoder
    print("\nCreating DualCLIPVisionEncoder...")
    encoder = DualCLIPVisionEncoder(config)
    
    # Create dummy input
    images = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = encoder(images, return_dict=True)
    
    print(f"Pooled output shape: {output['pooled_output'].shape}")
    print(f"Left branch output shape: {output['left_branch_output'].shape}")
    print(f"Right branch output shape: {output['right_branch_output'].shape}")
    print("\n✓ CLIP encoder works!\n")


def test_clip_with_pretrained():
    """Test loading pretrained CLIP weights."""
    print("=" * 60)
    print("Loading Pretrained CLIP Weights")
    print("=" * 60)
    
    try:
        print("\nAttempting to load openai/clip-vit-base-patch32...")
        
        # Use from_pretrained convenience method
        encoder = DualCLIPVisionEncoder.from_pretrained(
            'openai/clip-vit-base-patch32',
            output_mode='right'
        )
        
        # Test forward pass
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = encoder(images, return_dict=True)
        
        print(f"\n✓ Loaded pretrained CLIP weights!")
        print(f"Output shape: {output['pooled_output'].shape}")
        
    except Exception as e:
        print(f"\n✗ Could not load pretrained weights: {e}")
        print("This requires internet connection and transformers library")


def test_clip_with_text_features():
    """Simulate using text features as auxiliary tokens."""
    print("=" * 60)
    print("Using Text Features as Auxiliary Tokens")
    print("=" * 60)
    
    config = DualVisionConfig(
        encoder_type='clip',
        hidden_size=512,  # CLIP base uses 512
        num_layers=12,
        num_heads=8,
        image_size=224,
        patch_size=32,
    )
    
    encoder = DualCLIPVisionEncoder(config)
    
    # Simulate text features from CLIP text encoder
    # In practice, these would come from: clip_text_encoder(text_tokens)
    batch_size = 2
    seq_len = 50  # 49 patches + 1 CLS for patch_size=32
    
    images = torch.randn(batch_size, 3, 224, 224)
    text_features = torch.randn(batch_size, seq_len, 512)  # Simulated text features
    
    print(f"\nImage shape: {images.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    with torch.no_grad():
        output = encoder(images, auxiliary_tokens=text_features, return_dict=True)
    
    print(f"\nOutput shape: {output['pooled_output'].shape}")
    print("\n✓ Using text features as auxiliary tokens works!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DualCLIPVisionEncoder Test Suite")
    print("=" * 60 + "\n")
    
    test_clip_encoder()
    test_clip_with_pretrained()
    test_clip_with_text_features()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
