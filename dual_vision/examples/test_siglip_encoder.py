"""
Example: Testing Dual SigLIP Vision Encoder

This script demonstrates how to use the SigLIP-based dual vision encoder.
"""

import torch
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import DualVisionConfig
from modeling_siglip import DualSigLIPVisionEncoder


def test_siglip_encoder():
    """Test SigLIP vision encoder."""
    print("=" * 60)
    print("Testing DualSigLIPVisionEncoder")
    print("=" * 60)
    
    # Create config
    # SigLIP uses mean pooling and no CLS token
    config = DualVisionConfig(
        encoder_type='siglip',
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        image_size=224,
        patch_size=16,
        use_cls_token=False,
        pooling_type='mean',
        output_mode='right',
    )
    
    print("\nCreating DualSigLIPVisionEncoder...")
    encoder = DualSigLIPVisionEncoder(config)
    
    # Create dummy input
    images = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = encoder(images, return_dict=True)
    
    print(f"Pooled output shape (mean pooling): {output['pooled_output'].shape}")
    print(f"Left branch output shape: {output['left_branch_output'].shape}")
    print(f"Right branch output shape: {output['right_branch_output'].shape}")
    print("\n✓ SigLIP encoder works!\n")


def test_siglip_with_pretrained():
    """Test loading pretrained SigLIP weights."""
    print("=" * 60)
    print("Loading Pretrained SigLIP Weights")
    print("=" * 60)
    
    try:
        print("\nAttempting to load google/siglip-base-patch16-224...")
        
        # Use from_pretrained convenience method
        encoder = DualSigLIPVisionEncoder.from_pretrained(
            'google/siglip-base-patch16-224',
            output_mode='right'
        )
        
        # Test forward pass
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = encoder(images, return_dict=True)
        
        print(f"\n✓ Loaded pretrained SigLIP weights!")
        print(f"Output shape: {output['pooled_output'].shape}")
        
    except Exception as e:
        print(f"\n✗ Could not load pretrained weights: {e}")
        print("This requires internet connection and transformers library")


def test_siglip_no_cls_token():
    """Verify SigLIP doesn't use CLS token."""
    print("=" * 60)
    print("Verifying No CLS Token Usage")
    print("=" * 60)
    
    config = DualVisionConfig(
        encoder_type='siglip',
        image_size=224,
        patch_size=16,
    )
    
    encoder = DualSigLIPVisionEncoder(config)
    
    # Calculate expected sequence length
    num_patches = (224 // 16) ** 2  # 196 patches
    print(f"\nExpected sequence length (no CLS): {num_patches}")
    
    # Forward pass
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = encoder(images, return_dict=True)
    
    actual_seq_len = output['left_branch_output'].shape[1]
    print(f"Actual sequence length: {actual_seq_len}")
    
    assert actual_seq_len == num_patches, "SigLIP should not have CLS token!"
    print("\n✓ Confirmed: No CLS token in SigLIP!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DualSigLIPVisionEncoder Test Suite")
    print("=" * 60 + "\n")
    
    test_siglip_encoder()
    test_siglip_with_pretrained()
    test_siglip_no_cls_token()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
