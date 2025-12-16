"""
Example: Testing Dual ViT Encoder

This script demonstrates how to:
1. Create a dual ViT encoder
2. Load pretrained weights from HuggingFace
3. Perform forward pass with image input
4. Use random auxiliary tokens or custom tokens
"""

import torch
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import DualVisionConfig
from modeling_vit import DualViTEncoder


def test_basic_forward():
    """Test basic forward pass with random auxiliary tokens."""
    print("=" * 60)
    print("Test 1: Basic Forward Pass with Random Tokens")
    print("=" * 60)
    
    # Create config
    config = DualVisionConfig(
        encoder_type='vit',
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        image_size=224,
        patch_size=16,
        auxiliary_token_init='gaussian',
        random_token_std=0.02,
        output_mode='right',  # Use right branch output by default
    )
    
    # Create encoder
    print("\nCreating DualViTEncoder...")
    encoder = DualViTEncoder(config)
    
    # Print model info
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = encoder(images, return_dict=True)
    
    print(f"Pooled output shape: {output['pooled_output'].shape}")
    print(f"Left branch shape: {output['left_branch_output'].shape}")
    print(f"Right branch shape: {output['right_branch_output'].shape}")
    print("\n✓ Basic forward pass successful!\n")


def test_with_pretrained():
    """Test loading pretrained weights."""
    print("=" * 60)
    print("Test 2: Loading Pretrained Weights")
    print("=" * 60)
    
    # This requires internet connection and transformers library
    try:
        print("\nAttempting to load google/vit-base-patch16-224...")
        
        config = DualVisionConfig(
            encoder_type='vit',
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            image_size=224,
            patch_size=16,
        )
        
        encoder = DualViTEncoder(config)
        
        # Load pretrained weights
        encoder.load_pretrained_weights('google/vit-base-patch16-224', library='transformers')
        
        # Test forward pass
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = encoder(images)
        
        print(f"\n✓ Loaded pretrained weights successfully!")
        print(f"Output shape: {output[0].shape}")
        
    except Exception as e:
        print(f"\n✗ Could not load pretrained weights: {e}")
        print("This is expected if you don't have internet or transformers library")


def test_custom_auxiliary_tokens():
    """Test with custom auxiliary tokens instead of random."""
    print("=" * 60)
    print("Test 3: Custom Auxiliary Tokens")
    print("=" * 60)
    
    config = DualVisionConfig(
        encoder_type='vit',
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        image_size=224,
        patch_size=16,
    )
    
    encoder = DualViTEncoder(config)
    
    # Create custom auxiliary tokens (e.g., from text encoder or LLM hidden states)
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token for ViT
    hidden_dim = 768
    
    images = torch.randn(batch_size, 3, 224, 224)
    custom_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\nUsing custom auxiliary tokens: {custom_tokens.shape}")
    
    with torch.no_grad():
        output = encoder(images, auxiliary_tokens=custom_tokens, return_dict=True)
    
    print(f"Output shape: {output['pooled_output'].shape}")
    print("\n✓ Custom auxiliary tokens work!\n")


def test_parameter_freezing():
    """Test freezing left branch."""
    print("=" * 60)
    print("Test 4: Parameter Freezing")
    print("=" * 60)
    
    config = DualVisionConfig(encoder_type='vit')
    encoder = DualViTEncoder(config)
    
    # Count trainable params before freezing
    trainable_before = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTrainable params before freezing: {trainable_before:,}")
    
    # Freeze left branch
    encoder.freeze_left_branch()
    
    # Count trainable params after freezing
    trainable_after = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    left_params = sum(p.numel() for p in encoder.get_left_parameters())
    right_params = sum(p.numel() for p in encoder.get_right_parameters())
    
    print(f"Trainable params after freezing: {trainable_after:,}")
    print(f"Left branch params (frozen): {left_params:,}")
    print(f"Right branch params (trainable): {right_params:,}")
    print("\n✓ Parameter freezing works!\n")


def test_different_output_modes():
    """Test different output modes."""
    print("=" * 60)
    print("Test 5: Different Output Modes")
    print("=" * 60)
    
    images = torch.randn(2, 3, 224, 224)
    
    for mode in ['right', 'left', 'both', 'concat']:
        config = DualVisionConfig(
            encoder_type='vit',
            output_mode=mode
        )
        encoder = DualViTEncoder(config)
        
        with torch.no_grad():
            output = encoder(images, return_dict=True)
        
        pooled = output['pooled_output']
        if mode == 'both':
            print(f"\nMode '{mode}': tuple with shapes {pooled[0].shape}, {pooled[1].shape}")
        else:
            print(f"\nMode '{mode}': shape {pooled.shape}")
    
    print("\n✓ All output modes work!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DualViTEncoder Test Suite")
    print("=" * 60 + "\n")
    
    test_basic_forward()
    test_with_pretrained()
    test_custom_auxiliary_tokens()
    test_parameter_freezing()
    test_different_output_modes()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
