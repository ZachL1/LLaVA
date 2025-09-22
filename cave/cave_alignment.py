"""
CAVE Representation Alignment Module

This module provides a wrapper around the CAVE encoder for aligning representations
with pretrained vision models like DINOv2, CLIP, and SAM2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
import numpy as np
from omegaconf import DictConfig
import os
import json
import sys

from .cave_with_vae import CAVEWithVAE, CAVELatentEncoder, CAVELatentDecoder, SD35VAEWrapper


class SAM2ProjectorModule(nn.Module):
    """
    Special projector for SAM2 that handles token dimension transformation.
    Projects from CAVE's 256 tokens to SAM2's 4096 spatial tokens with 256 channels.
    """
    
    def __init__(self, cave_hidden_size: int, sam2_embed_dim: int = 256, sam2_num_tokens: int = 4096):
        super().__init__()
        self.cave_hidden_size = cave_hidden_size
        self.sam2_embed_dim = sam2_embed_dim
        self.sam2_num_tokens = sam2_num_tokens
        
        # Project embedding dimension from 1536 to 256 (SAM2 channels)
        self.embed_proj = nn.Linear(cave_hidden_size, sam2_embed_dim, bias=True)
        
        # Project token count from 256 to 4096 (SAM2 spatial tokens)
        self.token_proj = nn.Linear(256, sam2_num_tokens, bias=True)
        
        # Initialize projection weights
        nn.init.normal_(self.embed_proj.weight, std=0.02)
        nn.init.zeros_(self.embed_proj.bias)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.zeros_(self.token_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project CAVE embeddings to SAM2 format.
        
        Args:
            x: CAVE embeddings (B, 256, cave_hidden_size)
            
        Returns:
            SAM2-compatible embeddings (B, 4096, 256)
        """
        # Project embedding dimension: (B, 256, cave_hidden_size) -> (B, 256, sam2_embed_dim)
        x_proj = self.embed_proj(x)
        
        # Project token dimension: (B, 256, sam2_embed_dim) -> (B, 4096, sam2_embed_dim)
        x_proj = x_proj.transpose(1, 2)  # (B, 256, sam2_embed_dim) -> (B, sam2_embed_dim, 256)
        x_proj = self.token_proj(x_proj)  # (B, sam2_embed_dim, 256) -> (B, sam2_embed_dim, 4096)
        x_proj = x_proj.transpose(1, 2)  # (B, sam2_embed_dim, 4096) -> (B, 4096, sam2_embed_dim)
        
        return x_proj


# Import prompt embedding utilities
try:
    from utils.prompt_embeddings import PromptEmbeddingManager
    PROMPT_EMBEDDINGS_AVAILABLE = True
except ImportError:
    PROMPT_EMBEDDINGS_AVAILABLE = False
    print("Warning: Prompt embeddings utility not available")

# All target models are managed via HuggingFace Hub - no legacy imports needed
# Available models:
# - facebook/dinov2-base (DINOv2)
# - facebook/dinov3-vitb16-pretrain-lvd1689m (DINOv3)
# - openai/clip-vit-base-patch32 (CLIP)
# - facebook/sam2-hiera-base-plus (SAM2)

# HuggingFace imports for target model loading
from transformers import AutoModel, CLIPModel, AutoImageProcessor, CLIPProcessor


class CAVEAlignmentEncoder(nn.Module):
    """
    CAVE Encoder wrapper for representation alignment with pretrained vision models.
    
    Features:
    - Supports multiple target models (DINOv2, CLIP, SAM2)
    - Model-specific context tokens
    - Projectors for dimension alignment
    - Optional loading from reconstruction stage checkpoints
    """
    
    # Predefined model configurations
    MODEL_CONFIGS = {
        'dinov2_small': {
            'embed_dim': 384,
            'num_tokens_cls': 1,  # CLS token only
            'num_tokens_all': 256,  # All patch tokens (16x16 for 224x224 image with patch_size=14)
            'patch_size': 14,
            'image_size': 224,
        },
        'dinov2_base': {
            'embed_dim': 768,
            'num_tokens_cls': 1,
            'num_tokens_all': 256,
            'patch_size': 14,
            'image_size': 224,
        },
        'dinov2_large': {
            'embed_dim': 1024,
            'num_tokens_cls': 1,
            'num_tokens_all': 256,
            'patch_size': 14,
            'image_size': 224,
        },
        'dinov2_giant': {
            'embed_dim': 1536,
            'num_tokens_cls': 1,
            'num_tokens_all': 256,
            'patch_size': 14,
            'image_size': 224,
        },
        'dinov3_small': {
            'embed_dim': 384,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,  # 14x14 for 224x224 image with patch_size=16
            'patch_size': 16,
            'image_size': 224,
        },
        'dinov3_base': {
            'embed_dim': 768,
            'num_tokens_cls': 1,
            'num_tokens_all': 201,  # All 201 tokens (1 CLS + 200 patch tokens) as requested
            'patch_size': 16,
            'image_size': 224,
        },
        'dinov3_large': {
            'embed_dim': 1024,
            'num_tokens_cls': 1,
            'num_tokens_all': 201,  # All 201 tokens (1 CLS + 200 patch tokens) - consistent with dinov3_base
            'patch_size': 16,
            'image_size': 224,
        },
        'dinov3_giant': {
            'embed_dim': 1280,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,  # 14x14 for 224x224 image with patch_size=16
            'patch_size': 16,
            'image_size': 224,
        },
        'clip_vit_b16': {
            'embed_dim': 512,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,  # 14x14 for 224x224 image with patch_size=16
            'patch_size': 16,
            'image_size': 224,
        },
        'clip_vit_b32': {
            'embed_dim': 512,
            'num_tokens_cls': 1,
            'num_tokens_all': 49,   # 7x7 for 224x224 image with patch_size=32
            'patch_size': 32,
            'image_size': 224,
        },
        'clip_vit_l14': {
            'embed_dim': 768,
            'num_tokens_cls': 1,
            'num_tokens_all': 256,  # 16x16 for 224x224 image with patch_size=14
            'patch_size': 14,
            'image_size': 224,
        },
        'sam2_hiera_tiny': {
            'embed_dim': 96,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,  # Estimated for hierarchical vision transformer
            'patch_size': 16,
            'image_size': 224,
        },
        'sam2_hiera_small': {
            'embed_dim': 384,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,
            'patch_size': 16,
            'image_size': 224,
        },
        'sam2_hiera_base': {
            'embed_dim': 768,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,
            'patch_size': 16,
            'image_size': 224,
        },
        'sam2_hiera_large': {
            'embed_dim': 1024,
            'num_tokens_cls': 1,
            'num_tokens_all': 196,
            'patch_size': 16,
            'image_size': 224,
        },
        'reconstruction': {
            'embed_dim': None,  # Will be set to CAVE's hidden_size
            'num_tokens_cls': 16,   # Default reconstruction tokens for CLS-like mode
            'num_tokens_all': 256,  # All reconstruction tokens for all-tokens mode
            'patch_size': 2,
            'image_size': 256,
        },
        # Cleaned up model names for HuggingFace
        'clip_base': {
            'embed_dim': 768,  # CLIP ViT-Base vision model outputs 768-dim embeddings
            'num_tokens_cls': 1,
            'num_tokens_all': 49,  # 7x7 for 224x224 image with patch_size=32  
            'patch_size': 32,
            'image_size': 224,
        },
        'sam2_base': {
            'embed_dim': 4096,  # Target token dimension as requested
            'num_tokens_cls': 1,
            'num_tokens_all': 256,  # Target model token numbers as requested
            'patch_size': 16,  # Effective patch size for 224->14 spatial resolution
            'image_size': 224,
        },
    }
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.alignment_config = config.model.get('alignment', {})
        
        # Token mode: 'cls' for CLS tokens only, 'all' for all patch tokens
        self.token_mode = self.alignment_config.get('token_mode', 'cls')
        print(f"Alignment token mode: {self.token_mode}")
        
        # Initialize VAE following the same pattern as CAVEWithVAE
        vae_path = getattr(config.model, 'vae_path', None)
        if hasattr(config.model, 'sd3_model_path') and config.model.sd3_model_path:
            # Check if the path exists before using it
            if os.path.exists(config.model.sd3_model_path):
                vae_path = config.model.sd3_model_path
            else:
                print(f"Warning: SD3.5 model path not found: {config.model.sd3_model_path}")
                vae_path = None
            
        self.vae = SD35VAEWrapper(
            vae_path=vae_path,
            dtype=torch.float16,
            device=None
        )
        
        # Initialize CAVE encoder
        self.cave_encoder = CAVELatentEncoder(config)
        
        # Keep CAVE encoder in original precision (float32)
        
        # Get CAVE hidden size from config
        self.cave_hidden_size = config.model.encoder.get("hidden_size", 64 * config.model.encoder.get("depth", 12))
        
        # Set reconstruction embedding dimension
        self.MODEL_CONFIGS['reconstruction']['embed_dim'] = self.cave_hidden_size
        
        # Initialize prompt embedding manager if available
        self.prompt_embedding_manager = None
        self.use_prompt_embeddings = self.alignment_config.get('use_prompt_embeddings', True)
        
        if self.use_prompt_embeddings and PROMPT_EMBEDDINGS_AVAILABLE:
            prompt_embeddings_dir = self.alignment_config.get(
                'prompt_embeddings_dir', 
                '/mnt/bn/kgvanasgcparnold/annan/CAVE/embeddings/prompt_embeddings'
            )
            if os.path.exists(prompt_embeddings_dir):
                try:
                    self.prompt_embedding_manager = PromptEmbeddingManager(
                        embeddings_dir=prompt_embeddings_dir,
                        cave_hidden_size=self.cave_hidden_size,
                        device='cpu'  # Will be moved to correct device later
                    )
                    print(f"Initialized prompt embedding manager from {prompt_embeddings_dir}")
                except Exception as e:
                    print(f"Warning: Failed to initialize prompt embedding manager: {e}")
                    self.use_prompt_embeddings = False
            else:
                print(f"Warning: Prompt embeddings directory not found: {prompt_embeddings_dir}")
                self.use_prompt_embeddings = False
        else:
            print("Prompt embeddings disabled or not available")
        
        # Target models to support
        self.target_models = self.alignment_config.get('target_models', ['reconstruction'])
        
        # Initialize model-specific components
        self._init_model_specific_components()
        
        # Default model selection
        self.current_model = 'reconstruction'
        
    def _init_model_specific_components(self):
        """Initialize model-specific context tokens and projectors."""
        
        self.model_context_tokens = nn.ModuleDict()
        self.model_projectors = nn.ModuleDict()
        
        for model_name in self.target_models:
            if model_name not in self.MODEL_CONFIGS:
                raise ValueError(f"Unsupported model: {model_name}")
                
            model_config = self.MODEL_CONFIGS[model_name]
            
            # Choose number of tokens based on mode
            if self.token_mode == 'cls':
                num_tokens = model_config['num_tokens_cls']
            elif self.token_mode == 'all':
                num_tokens = model_config['num_tokens_all']
            else:
                raise ValueError(f"Unsupported token mode: {self.token_mode}")
            
            print(f"Model {model_name}: Creating {num_tokens} context tokens for {self.token_mode} mode")
            
            scale = self.cave_hidden_size ** -0.5
            
            # Create a simple module to hold the context tokens
            context_module = nn.Module()
            context_module.tokens = nn.Parameter(
                scale * torch.randn(num_tokens, self.cave_hidden_size, dtype=torch.float32)
            )
            self.model_context_tokens[model_name] = context_module
            
            # Create model-specific component (projector for alignment, decoder for reconstruction)
            if model_name == 'reconstruction':
                # For reconstruction, create a decoder using the decoder config
                projector = CAVELatentDecoder(self.config)
                print(f"Created decoder for reconstruction training")
            elif model_name == 'sam2_base':
                # Special handling for SAM2 - use custom projector for token shape transformation
                projector = SAM2ProjectorModule(
                    cave_hidden_size=self.cave_hidden_size,
                    sam2_embed_dim=256,  # SAM2 channel dimension
                    sam2_num_tokens=model_config['embed_dim']  # SAM2 spatial tokens (4096)
                )
                print(f"Created SAM2 projector for {model_name} (256 tokens -> {model_config['embed_dim']} tokens, {self.cave_hidden_size} -> 256)")
            else:
                # For other alignment models, create projector for dimension alignment
                target_dim = model_config['embed_dim']
                if target_dim != self.cave_hidden_size:
                    # Simple single linear layer projection
                    projector = nn.Linear(self.cave_hidden_size, target_dim, bias=True)
                    # Initialize with small weights
                    nn.init.normal_(projector.weight, std=0.02)
                    nn.init.zeros_(projector.bias)
                else:
                    projector = nn.Identity()
                print(f"Created projector for {model_name} alignment (dim: {self.cave_hidden_size} -> {target_dim})")
            
            self.model_projectors[model_name] = projector
    
    def _move_prompt_embeddings_to_device(self, device):
        """Move prompt embedding manager to specified device."""
        if self.prompt_embedding_manager is not None:
            self.prompt_embedding_manager.to(device)
    
    def to(self, device):
        """Override to method to handle prompt embeddings."""
        # Call parent to() method
        result = super().to(device)
        
        # Move prompt embeddings to device
        self._move_prompt_embeddings_to_device(device)
        
        return result
    
    def cuda(self, device=None):
        """Override cuda method to handle prompt embeddings."""
        result = super().cuda(device)
        self._move_prompt_embeddings_to_device('cuda' if device is None else device)
        return result
            
    def load_reconstruction_weights(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained weights from reconstruction stage.
        Supports both single file checkpoints and DeepSpeed checkpoint directories.
        
        Args:
            checkpoint_path: Path to the reconstruction checkpoint (file or directory)
            strict: Whether to enforce strict loading
        """
        print(f"Loading reconstruction weights from {checkpoint_path}")
        
        # Check if it's a DeepSpeed checkpoint directory
        if os.path.isdir(checkpoint_path):
            # DeepSpeed checkpoint format
            try:
                from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
                print("Loading DeepSpeed checkpoint...")
                state_dict = load_state_dict_from_zero_checkpoint(checkpoint_path)
            except ImportError:
                print("DeepSpeed not available, trying manual loading...")
                # Manual loading for DeepSpeed checkpoints
                import glob
                import json
                
                # Look for the latest checkpoint files
                mp_rank_files = glob.glob(os.path.join(checkpoint_path, "mp_rank_*_model_states.pt"))
                if not mp_rank_files:
                    raise FileNotFoundError(f"No DeepSpeed checkpoint files found in {checkpoint_path}")
                
                # Load and merge all rank files
                state_dict = {}
                for rank_file in mp_rank_files:
                    try:
                        rank_state = torch.load(rank_file, map_location='cpu', weights_only=False)
                    except TypeError:
                        rank_state = torch.load(rank_file, map_location='cpu')
                    
                    if 'module' in rank_state:
                        rank_state = rank_state['module']
                    
                    # Merge state dicts
                    for key, value in rank_state.items():
                        if key not in state_dict:
                            state_dict[key] = value
        else:
            # Single file checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint
            
        # Filter encoder weights
        encoder_weights = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                encoder_weights[new_key] = value
            elif key.startswith('cave_encoder.'):
                new_key = key.replace('cave_encoder.', '')
                encoder_weights[new_key] = value
            elif key.startswith('module.encoder.'):
                new_key = key.replace('module.encoder.', '')
                encoder_weights[new_key] = value
            elif key.startswith('module.cave_encoder.'):
                new_key = key.replace('module.cave_encoder.', '')
                encoder_weights[new_key] = value
                
        if not encoder_weights:
            print("Warning: No encoder weights found in checkpoint")
            print(f"Available keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
            
        # Load weights
        missing_keys, unexpected_keys = self.cave_encoder.load_state_dict(
            encoder_weights, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        print("Successfully loaded reconstruction weights")
        
    def load_and_freeze_decoder(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained decoder weights and freeze them for alignment training.
        This allows using a pre-trained decoder while only training alignment projectors.
        
        Args:
            checkpoint_path: Path to the checkpoint containing decoder weights
            strict: Whether to enforce strict loading
        """
        print(f"Loading and freezing decoder weights from {checkpoint_path}")
        
        # Check if reconstruction decoder exists
        if 'reconstruction' not in self.model_projectors:
            print("Warning: No reconstruction decoder found. Creating one...")
            decoder = CAVELatentDecoder(self.config)
            self.model_projectors['reconstruction'] = decoder
        
        decoder = self.model_projectors['reconstruction']
        
        # Load checkpoint efficiently
        state_dict = {}
        
        if os.path.isdir(checkpoint_path):
            # Check for converted_weights directory first
            converted_dir = os.path.join(checkpoint_path, "converted_weights")
            if os.path.exists(converted_dir):
                print(f"Found converted weights directory: {converted_dir}")
                
                # Load from sharded checkpoint
                index_file = os.path.join(converted_dir, "pytorch_model.bin.index.json")
                if os.path.exists(index_file):
                    print("Loading from sharded checkpoint...")
                    import json
                    
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    
                    # Find unique shard files containing decoder weights
                    weight_map = index_data['weight_map']
                    decoder_shards = set()
                    for param_name, file_name in weight_map.items():
                        if param_name.startswith('decoder.'):
                            decoder_shards.add(file_name)
                    
                    print(f"Loading decoder from {len(decoder_shards)} shard(s)...")
                    
                    # Always load to CPU first for memory efficiency, move to target device later
                    map_location = 'cpu'
                    
                    # Load each relevant shard efficiently
                    for shard_file in decoder_shards:
                        shard_path = os.path.join(converted_dir, shard_file)
                        
                        shard_data = torch.load(shard_path, map_location=map_location, weights_only=True)
                        
                        for key, value in shard_data.items():
                            if key.startswith('decoder.'):
                                # Remove 'decoder.mmdit.' or 'decoder.' prefix
                                if key.startswith('decoder.mmdit.'):
                                    decoder_key = key.replace('decoder.mmdit.', '', 1)
                                else:
                                    decoder_key = key.replace('decoder.', '', 1)
                                state_dict[decoder_key] = value
                        
                        # Free memory immediately
                        del shard_data
                        import gc; gc.collect()
                        
                    print(f"Loaded {len(state_dict)} decoder parameters")
                else:
                    raise FileNotFoundError(f"Index file not found: {index_file}")
            else:
                raise FileNotFoundError(f"Converted weights directory not found: {converted_dir}")
        else:
            # Single file checkpoint
            print(f"Loading single file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            
            if 'model' in checkpoint:
                full_state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            else:
                full_state_dict = checkpoint
            
            # Extract decoder weights and process keys properly
            for key, value in full_state_dict.items():
                if key.startswith('decoder.'):
                    # Remove 'decoder.mmdit.' or 'decoder.' prefix and prepare for mmdit loading
                    if key.startswith('decoder.mmdit.'):
                        decoder_key = key.replace('decoder.mmdit.', '', 1)
                    else:
                        decoder_key = key.replace('decoder.', '', 1)
                    state_dict[decoder_key] = value
        
        # Load decoder weights into the mmdit submodule
        if state_dict:
            # Add mmdit prefix to all keys since we're loading into the decoder wrapper
            prefixed_state_dict = {}
            for key, value in state_dict.items():
                prefixed_key = f"mmdit.{key}"
                prefixed_state_dict[prefixed_key] = value
            
            # Load into the decoder wrapper (which will route to mmdit submodule)
            missing_keys, unexpected_keys = decoder.load_state_dict(prefixed_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing decoder keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing decoder keys: {missing_keys}")
                
            if unexpected_keys:
                print(f"Unexpected decoder keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected decoder keys: {unexpected_keys}")
                
            # Check if loading was successful (some weights loaded)
            loaded_params = len(state_dict) - len(unexpected_keys)
            if loaded_params > 0:
                print(f"✓ Successfully loaded {loaded_params} decoder parameters into mmdit")
            else:
                print("⚠ Warning: No matching decoder parameters were loaded into mmdit")
                
        else:
            print("Warning: No decoder weights found in checkpoint")
            return
        
        # Freeze decoder parameters
        frozen_params = 0
        for param in decoder.parameters():
            param.requires_grad = False
            frozen_params += 1
        
        decoder.eval()  # Set to evaluation mode
        
        # Ensure decoder is on the same device as the main model for fast inference
        main_device = next(self.parameters()).device
        decoder_device = next(decoder.parameters()).device
        if decoder_device != main_device:
            print(f"Moving frozen decoder from {decoder_device} to {main_device} for fast inference...")
            decoder = decoder.to(main_device)
        
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"✓ Successfully loaded and froze decoder ({total_params:,} parameters)")
        
    def load_and_freeze_decoder_optimized(self, checkpoint_path: str, accelerator, strict: bool = False):
        """
        Optimized decoder loading for multi-GPU training.
        Only loads on main process, then synchronizes across all processes.
        
        Args:
            checkpoint_path: Path to the checkpoint containing decoder weights
            accelerator: HuggingFace Accelerator instance
            strict: Whether to enforce strict loading
        """
        # Check if reconstruction decoder exists
        if 'reconstruction' not in self.model_projectors:
            if accelerator.is_main_process:
                print("Warning: No reconstruction decoder found. Creating one...")
            decoder = CAVELatentDecoder(self.config)
            self.model_projectors['reconstruction'] = decoder
        
        decoder = self.model_projectors['reconstruction']
        
        # Load checkpoint only on main process
        state_dict = {}
        
        if accelerator.is_main_process:
            print(f"Loading and freezing decoder weights from {checkpoint_path}")
            
            if os.path.isdir(checkpoint_path):
                # Check for converted_weights directory first
                converted_dir = os.path.join(checkpoint_path, "converted_weights")
                if os.path.exists(converted_dir):
                    print(f"Found converted weights directory: {converted_dir}")
                    
                    # Load from sharded checkpoint
                    index_file = os.path.join(converted_dir, "pytorch_model.bin.index.json")
                    if os.path.exists(index_file):
                        print("Loading from sharded checkpoint...")
                        import json
                        
                        with open(index_file, 'r') as f:
                            index_data = json.load(f)
                        
                        # Find unique shard files containing decoder weights
                        weight_map = index_data['weight_map']
                        decoder_shards = set()
                        for param_name, file_name in weight_map.items():
                            if param_name.startswith('decoder.'):
                                decoder_shards.add(file_name)
                        
                        print(f"Loading decoder from {len(decoder_shards)} shard(s)...")
                        
                        # Always load to CPU first for memory efficiency
                        map_location = 'cpu'
                        
                        # Load each relevant shard efficiently
                        for shard_file in decoder_shards:
                            shard_path = os.path.join(converted_dir, shard_file)
                            
                            shard_data = torch.load(shard_path, map_location=map_location, weights_only=True)
                            
                            for key, value in shard_data.items():
                                if key.startswith('decoder.'):
                                    # Remove 'decoder.mmdit.' or 'decoder.' prefix
                                    if key.startswith('decoder.mmdit.'):
                                        decoder_key = key.replace('decoder.mmdit.', '', 1)
                                    else:
                                        decoder_key = key.replace('decoder.', '', 1)
                                    state_dict[decoder_key] = value
                            
                            # Free memory immediately
                            del shard_data
                            import gc; gc.collect()
                            
                        print(f"Loaded {len(state_dict)} decoder parameters on main process")
                    else:
                        raise FileNotFoundError(f"Index file not found: {index_file}")
                else:
                    raise FileNotFoundError(f"Converted weights directory not found: {converted_dir}")
            else:
                # Single file checkpoint
                print(f"Loading single file: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                
                if 'model' in checkpoint:
                    full_state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    full_state_dict = checkpoint['state_dict']
                else:
                    full_state_dict = checkpoint
                
                # Extract decoder weights
                for key, value in full_state_dict.items():
                    if key.startswith('decoder.'):
                        if key.startswith('decoder.mmdit.'):
                            decoder_key = key.replace('decoder.mmdit.', '', 1)
                        else:
                            decoder_key = key.replace('decoder.', '', 1)
                        state_dict[decoder_key] = value
                
                print(f"Loaded {len(state_dict)} decoder parameters from single file")
        
        # Synchronize all processes before loading weights
        accelerator.wait_for_everyone()
        
        # Load state dict on all processes (main process has the actual weights, others get empty dict)
        if state_dict:  # Only main process has weights
            # Add mmdit prefix to all keys since we're loading into the decoder wrapper
            prefixed_state_dict = {}
            for key, value in state_dict.items():
                prefixed_key = f"mmdit.{key}"
                prefixed_state_dict[prefixed_key] = value
            
            # Load weights into decoder (with prefixed keys for mmdit submodule)
            missing_keys, unexpected_keys = decoder.load_state_dict(prefixed_state_dict, strict=strict)
            
            if missing_keys and accelerator.is_main_process:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys and accelerator.is_main_process:
                print(f"Unexpected keys: {unexpected_keys}")
        
        # Synchronize again after loading
        accelerator.wait_for_everyone()
        
        # Freeze decoder parameters on all processes
        frozen_params = 0
        for param in decoder.parameters():
            param.requires_grad = False
            frozen_params += 1
        decoder.eval()
        
        if accelerator.is_main_process:
            print(f"Decoder frozen with {frozen_params} parameters")
            print("Successfully loaded and froze decoder weights")

    def unfreeze_decoder(self):
        """Unfreeze decoder parameters (for reconstruction training)."""
        if 'reconstruction' in self.model_projectors:
            decoder = self.model_projectors['reconstruction']
            for param in decoder.parameters():
                param.requires_grad = True
            decoder.train()
            print("Decoder parameters unfrozen")
        else:
            print("Warning: No reconstruction decoder found to unfreeze")

    def set_target_model(self, model_name: str):
        """Set the current target model for alignment."""
        if model_name not in self.target_models:
            raise ValueError(f"Model {model_name} not in supported models: {self.target_models}")
        self.current_model = model_name
        
    def get_model_info(self, model_name: Optional[str] = None) -> Dict:
        """Get configuration info for a model."""
        if model_name is None:
            model_name = self.current_model
        
        config = self.MODEL_CONFIGS[model_name].copy()
        # Add current token count based on mode
        if self.token_mode == 'cls':
            config['num_tokens'] = config['num_tokens_cls']
        elif self.token_mode == 'all':
            config['num_tokens'] = config['num_tokens_all']
        
        return config
        
    def forward(self, x: torch.Tensor, target_model: Optional[str] = None, 
                return_raw_context: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with model-specific context tokens.
        
        Args:
            x: Input images (B, C, H, W)
            target_model: Target model name. If None, uses current_model
            return_raw_context: Whether to return raw context before projection
            
        Returns:
            Dictionary containing:
            - 'aligned_embedding': Projected embedding for target model
            - 'raw_context': Raw context from CAVE encoder (if requested)
            - 'model_name': Name of target model used
        """
        if target_model is None:
            target_model = self.current_model
            
        # Encode image to latent space
        with torch.no_grad():
            # Ensure input matches VAE precision (float16)
            x_vae = x.half() if x.dtype != torch.float16 else x
            x_latent = self.vae.encode(x_vae)
            # Convert VAE output from float16 to float32 for CAVE encoder
            x_latent = x_latent.float()
            
        # Get model-specific context tokens
        base_context_tokens = self.model_context_tokens[target_model].tokens
        batch_size = x_latent.shape[0]
        
        # Apply prompt embeddings if available
        if self.use_prompt_embeddings and self.prompt_embedding_manager is not None:
            # Create enhanced context tokens with prompt information
            try:
                enhanced_tokens = self.prompt_embedding_manager.add_prompt_to_context_tokens(
                    base_context_tokens, target_model
                )
                print("checking: ", target_model, enhanced_tokens)
                context = enhanced_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            except Exception as e:
                print(f"Warning: Failed to add prompt embeddings for {target_model}: {e}")
                # Fallback to base context tokens
                context = base_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Use base context tokens without prompt enhancement
            context = base_context_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Forward through CAVE encoder with specific context
        raw_context = self.cave_encoder(x_latent, context=context)
        
        # Handle reconstruction vs alignment differently
        if target_model == 'reconstruction':
            # For reconstruction, follow the original CAVE approach:
            # 1. raw_context are the context tokens from encoder
            # 2. decoder takes noise latent + context → reconstructed latent
            # 3. Compare reconstructed latent vs original latent (no VAE decode needed)
            
            decoder = self.model_projectors[target_model]
            
            # Create noise latent with same shape as input latent
            noise_latent = torch.randn_like(x_latent)
            
            # Decoder reconstructs latent from noise + context tokens
            reconstructed_latents = decoder(noise_latent, context=raw_context)
            
            # Return reconstructed latents for comparison with original latents
            # Store original latent for loss computation in training
            results = {
                'aligned_embedding': reconstructed_latents,  # Reconstructed latents
                'model_name': target_model,
                'original_latent': x_latent  # Original latent for loss
            }
            
            if return_raw_context:
                results['raw_context'] = raw_context
                
            return results
        else:
            # For alignment, use projector to align embedding dimensions
            projector = self.model_projectors[target_model]
            aligned_embedding = projector(raw_context)
        
        results = {
            'aligned_embedding': aligned_embedding,
            'model_name': target_model
        }
        
        if return_raw_context:
            results['raw_context'] = raw_context
            
        return results
        
    def encode_for_model(self, x: torch.Tensor, model_name: str) -> torch.Tensor:
        """
        Encode image for a specific target model.
        
        Args:
            x: Input images (B, C, H, W)
            model_name: Target model name
            
        Returns:
            Aligned embeddings (B, num_tokens, embed_dim)
        """
        results = self.forward(x, target_model=model_name)
        return results['aligned_embedding']
        
    def encode_batch_for_models(self, x: torch.Tensor, 
                               model_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode images for multiple target models.
        
        Args:
            x: Input images (B, C, H, W)
            model_names: List of target model names
            
        Returns:
            Dictionary mapping model names to aligned embeddings
        """
        results = {}
        for model_name in model_names:
            results[model_name] = self.encode_for_model(x, model_name)
        return results
    
    def forward_multi_task(self, x: torch.Tensor, 
                          task_models: List[str],
                          return_raw_context: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Multi-task forward pass for joint training.
        
        Args:
            x: Input images (B, C, H, W)
            task_models: List of target model names to train jointly
            return_raw_context: Whether to return raw context
            
        Returns:
            Dictionary mapping task names to their outputs:
            {
                'dinov2_base': {'aligned_embedding': ..., 'model_name': ...},
                'clip_base': {'aligned_embedding': ..., 'model_name': ...},
                'reconstruction': {'aligned_embedding': ..., 'original_latent': ...},
                ...
            }
        """
        results = {}
        
        for task_model in task_models:
            # Forward pass for each task
            task_output = self.forward(x, target_model=task_model, return_raw_context=return_raw_context)
            results[task_model] = task_output
            
        return results
        
    def save_checkpoint(self, output_dir, step=None, save_modular=True):
        """Save the alignment model checkpoint with modular components."""
        os.makedirs(output_dir, exist_ok=True)
        
        if save_modular:
            # Save CAVE encoder separately for reuse
            cave_encoder_checkpoint = {
                'cave_encoder_state_dict': self.cave_encoder.state_dict(),
                'hidden_size': self.cave_hidden_size,
            }
            
            # Build projection dimensions from actual projectors
            projection_dims = {}
            for name, proj in self.model_projectors.items():
                if hasattr(proj, 'out_features'):
                    projection_dims[name] = proj.out_features
                else:
                    # Identity projector case
                    projection_dims[name] = self.cave_hidden_size
            
            # Check if we have a reconstruction decoder
            has_reconstruction_decoder = 'reconstruction' in self.model_projectors
            has_alignment_projectors = any(task != 'reconstruction' for task in self.model_projectors.keys())
            
            if has_reconstruction_decoder and has_alignment_projectors:
                # Multi-task training: save encoder, decoder, AND alignment projectors
                decoder_checkpoint = {
                    'decoder_state_dict': self.model_projectors['reconstruction'].state_dict(),
                    'config': self.config,
                    'hidden_size': self.cave_hidden_size
                }
                
                # Save alignment projectors separately (excluding reconstruction)
                alignment_projectors = {name: proj for name, proj in self.model_projectors.items() if name != 'reconstruction'}
                projectors_checkpoint = {
                    'target_models': [name for name in self.target_models if name != 'reconstruction'],
                    'projection_dims': {name: proj.out_features if hasattr(proj, 'out_features') else self.cave_hidden_size 
                                      for name, proj in alignment_projectors.items()},
                    'projectors_state_dict': {name: proj.state_dict() for name, proj in alignment_projectors.items()},
                    'use_prompt_embeddings': self.use_prompt_embeddings,
                    'prompt_embedding_projection': self.prompt_embedding_manager.get_projection_state_dict() 
                                                 if self.prompt_embedding_manager is not None else None
                }
                
                if step is not None:
                    cave_path = os.path.join(output_dir, f'cave_encoder_step_{step}.pt')
                    decoder_path = os.path.join(output_dir, f'cave_decoder_step_{step}.pt')
                    projectors_path = os.path.join(output_dir, f'alignment_projectors_step_{step}.pt')
                else:
                    cave_path = os.path.join(output_dir, 'cave_encoder.pt')
                    decoder_path = os.path.join(output_dir, 'cave_decoder.pt')
                    projectors_path = os.path.join(output_dir, 'alignment_projectors.pt')
                
                torch.save(cave_encoder_checkpoint, cave_path)
                torch.save(decoder_checkpoint, decoder_path)
                torch.save(projectors_checkpoint, projectors_path)
                print(f"CAVE encoder saved to {cave_path}")
                print(f"CAVE decoder saved to {decoder_path}")
                print(f"Alignment projectors saved to {projectors_path}")
                
                return cave_path, decoder_path, projectors_path
                
            elif has_reconstruction_decoder:
                # For reconstruction training, save encoder and decoder separately
                decoder_checkpoint = {
                    'decoder_state_dict': self.model_projectors['reconstruction'].state_dict(),
                    'config': self.config,
                    'hidden_size': self.cave_hidden_size
                }
                
                if step is not None:
                    cave_path = os.path.join(output_dir, f'cave_encoder_step_{step}.pt')
                    decoder_path = os.path.join(output_dir, f'cave_decoder_step_{step}.pt')
                else:
                    cave_path = os.path.join(output_dir, 'cave_encoder.pt')
                    decoder_path = os.path.join(output_dir, 'cave_decoder.pt')
                
                torch.save(cave_encoder_checkpoint, cave_path)
                torch.save(decoder_checkpoint, decoder_path)
                print(f"CAVE encoder saved to {cave_path}")
                print(f"CAVE decoder saved to {decoder_path}")
                
                return cave_path, decoder_path
            else:
                # For alignment training, save target-specific projectors separately
                projectors_checkpoint = {
                    'target_models': self.target_models,
                    'projection_dims': projection_dims,
                    'projectors_state_dict': {name: proj.state_dict() 
                                            for name, proj in self.model_projectors.items()},
                    'use_prompt_embeddings': self.use_prompt_embeddings,
                    'prompt_embedding_projection': self.prompt_embedding_manager.get_projection_state_dict() 
                                                 if self.prompt_embedding_manager is not None else None
                }
                
                if step is not None:
                    cave_path = os.path.join(output_dir, f'cave_encoder_step_{step}.pt')
                    projectors_path = os.path.join(output_dir, f'projectors_step_{step}.pt')
                else:
                    cave_path = os.path.join(output_dir, 'cave_encoder.pt')
                    projectors_path = os.path.join(output_dir, 'projectors.pt')
                
                torch.save(cave_encoder_checkpoint, cave_path)
                torch.save(projectors_checkpoint, projectors_path)
                print(f"CAVE encoder saved to {cave_path}")
                print(f"Projectors saved to {projectors_path}")
                
                return cave_path, projectors_path
        else:
            # Build projection dimensions from actual projectors
            projection_dims = {}
            for name, proj in self.model_projectors.items():
                if hasattr(proj, 'out_features'):
                    projection_dims[name] = proj.out_features
                else:
                    # Identity projector case
                    projection_dims[name] = self.cave_hidden_size
            
            # Legacy unified checkpoint
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'target_models': self.target_models,
                'projection_dims': projection_dims,
                'hidden_size': self.cave_hidden_size,
            }
            
            if step is not None:
                checkpoint['step'] = step
                checkpoint_path = os.path.join(output_dir, f'alignment_checkpoint_step_{step}.pt')
            else:
                checkpoint_path = os.path.join(output_dir, 'alignment_checkpoint.pt')
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Unified checkpoint saved to {checkpoint_path}")
            return checkpoint_path
        
    def load_checkpoint(self, path: str, load_optimizer: bool = False, 
                       load_scheduler: bool = False) -> Dict:
        """Load training checkpoint."""
        print(f"Loading checkpoint from {path}")
        
        # Handle PyTorch 2.6+ weights_only parameter
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(path, map_location='cpu')
        
        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        
        result = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'loss': checkpoint.get('loss', None)
        }
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            result['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
            
        print("Checkpoint loaded successfully")
        return result
    
    def load_multi_task_checkpoint(self, checkpoint_dir: str):
        """
        Load multi-task checkpoint with all components: encoder, decoder, and alignment projectors.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint files
        """
        print(f"Loading multi-task checkpoint from {checkpoint_dir}")
        
        # Find the checkpoint files
        encoder_path = None
        decoder_path = None
        projectors_path = None
        
        # Look for step-specific files first, then fallback to generic names
        for file in os.listdir(checkpoint_dir):
            if file.startswith('cave_encoder_step_') and file.endswith('.pt'):
                encoder_path = os.path.join(checkpoint_dir, file)
            elif file.startswith('cave_decoder_step_') and file.endswith('.pt'):
                decoder_path = os.path.join(checkpoint_dir, file)
            elif file.startswith('alignment_projectors_step_') and file.endswith('.pt'):
                projectors_path = os.path.join(checkpoint_dir, file)
            elif file == 'cave_encoder.pt':
                encoder_path = os.path.join(checkpoint_dir, file)
            elif file == 'cave_decoder.pt':
                decoder_path = os.path.join(checkpoint_dir, file)
            elif file == 'alignment_projectors.pt':
                projectors_path = os.path.join(checkpoint_dir, file)
        
        # Load encoder
        if encoder_path and os.path.exists(encoder_path):
            encoder_checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=True)
            self.cave_encoder.load_state_dict(encoder_checkpoint['cave_encoder_state_dict'])
            print(f"✓ Encoder loaded from {encoder_path}")
        else:
            print("⚠️ Encoder checkpoint not found")
        
        # Load decoder
        if decoder_path and os.path.exists(decoder_path):
            decoder_checkpoint = torch.load(decoder_path, map_location='cpu', weights_only=True)
            if 'reconstruction' in self.model_projectors:
                # Add mmdit prefix to decoder state dict keys
                decoder_state_dict = decoder_checkpoint['decoder_state_dict']
                prefixed_decoder_state_dict = {}
                for key, value in decoder_state_dict.items():
                    prefixed_key = f"mmdit.{key}"
                    prefixed_decoder_state_dict[prefixed_key] = value
                
                # Load into the decoder wrapper (which will route to mmdit submodule)
                self.model_projectors['reconstruction'].load_state_dict(prefixed_decoder_state_dict)
                print(f"✓ Decoder loaded from {decoder_path}")
            else:
                print("⚠️ No reconstruction decoder in model to load into")
        else:
            print("⚠️ Decoder checkpoint not found")
        
        # Load alignment projectors
        if projectors_path and os.path.exists(projectors_path):
            projectors_checkpoint = torch.load(projectors_path, map_location='cpu', weights_only=True)
            projectors_state_dict = projectors_checkpoint['projectors_state_dict']
            
            loaded_projectors = 0
            for name, state_dict in projectors_state_dict.items():
                if name in self.model_projectors:
                    self.model_projectors[name].load_state_dict(state_dict)
                    loaded_projectors += 1
                    print(f"✓ Projector '{name}' loaded")
                else:
                    print(f"⚠️ Projector '{name}' not found in model")
            
            print(f"✓ {loaded_projectors} alignment projectors loaded from {projectors_path}")
        else:
            print("⚠️ Alignment projectors checkpoint not found")
        
        print("Multi-task checkpoint loading completed")
    
    @classmethod
    def load_checkpoint_legacy(cls, checkpoint_path, target_models, projection_dims, vae_model_path, device='cuda'):
        """Load unified checkpoint (legacy method)."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        hidden_size = checkpoint.get('hidden_size', 1536)
        model = cls(
            target_models=checkpoint['target_models'],
            projection_dims=checkpoint['projection_dims'],
            vae_model_path=vae_model_path,
            hidden_size=hidden_size,
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Unified checkpoint loaded from {checkpoint_path}")
        
        step = checkpoint.get('step', None)
        return model, step
    
    @classmethod
    def load_cave_encoder(cls, cave_encoder_path, target_models, projection_dims, vae_model_path, device='cuda'):
        """Load only the CAVE encoder from modular checkpoint."""
        checkpoint = torch.load(cave_encoder_path, map_location=device)
        
        hidden_size = checkpoint.get('hidden_size', 1536)
        model = cls(
            target_models=target_models,
            projection_dims=projection_dims,
            vae_model_path=vae_model_path,
            hidden_size=hidden_size,
            device=device
        )
        
        # Load only the CAVE encoder weights
        model.cave_encoder.load_state_dict(checkpoint['cave_encoder_state_dict'])
        print(f"CAVE encoder loaded from {cave_encoder_path}")
        
        return model
    
    def load_projectors(self, projectors_path):
        """Load projectors from modular checkpoint into existing model."""
        # Handle PyTorch 2.6+ weights_only parameter
        try:
            checkpoint = torch.load(projectors_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(projectors_path, map_location='cpu')
        
        # Verify compatibility
        if checkpoint['target_models'] != self.target_models:
            print(f"Warning: Target models mismatch. Checkpoint: {checkpoint['target_models']}, Current: {self.target_models}")
        
        # Build current projection dimensions from actual projectors
        current_projection_dims = {}
        for name, proj in self.model_projectors.items():
            if hasattr(proj, 'out_features'):
                current_projection_dims[name] = proj.out_features
            else:
                # Identity projector case
                current_projection_dims[name] = self.cave_hidden_size
        
        if checkpoint['projection_dims'] != current_projection_dims:
            print(f"Warning: Projection dims mismatch. Checkpoint: {checkpoint['projection_dims']}, Current: {current_projection_dims}")
        
        # Load projector weights
        for name, state_dict in checkpoint['projectors_state_dict'].items():
            if name in self.model_projectors:
                self.model_projectors[name].load_state_dict(state_dict)
                print(f"Loaded projector weights for {name}")
            else:
                print(f"Warning: Projector {name} not found in current model")
        
        # Load prompt embedding projection weights if available
        if ('prompt_embedding_projection' in checkpoint and 
            checkpoint['prompt_embedding_projection'] is not None and
            self.prompt_embedding_manager is not None):
            try:
                self.prompt_embedding_manager.load_projection_state_dict(
                    checkpoint['prompt_embedding_projection']
                )
            except Exception as e:
                print(f"Warning: Failed to load prompt embedding projection: {e}")
        
        print(f"Projectors loaded from {projectors_path}")
    
    @classmethod
    def load_modular_checkpoint(cls, cave_encoder_path, projectors_path, target_models, projection_dims, vae_model_path, device='cuda'):
        """Load modular checkpoint with separate CAVE encoder and projectors."""
        # First load the CAVE encoder
        model = cls.load_cave_encoder(cave_encoder_path, target_models, projection_dims, vae_model_path, device)
        
        # Then load the projectors
        model.load_projectors(projectors_path)
        
        print(f"Modular checkpoint loaded: encoder from {cave_encoder_path}, projectors from {projectors_path}")
        return model
        
    def freeze_cave_encoder(self):
        """Freeze CAVE encoder parameters (useful for projector-only training)."""
        for param in self.cave_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_cave_encoder(self):
        """Unfreeze CAVE encoder parameters."""
        for param in self.cave_encoder.parameters():
            param.requires_grad = True
            
    def freeze_model_components(self, model_name: str):
        """Freeze components for a specific model."""
        if model_name in self.model_context_tokens:
            for param in self.model_context_tokens[model_name].parameters():
                param.requires_grad = False
        if model_name in self.model_projectors:
            for param in self.model_projectors[model_name].parameters():
                param.requires_grad = False
                
    def unfreeze_model_components(self, model_name: str):
        """Unfreeze components for a specific model."""
        if model_name in self.model_context_tokens:
            for param in self.model_context_tokens[model_name].parameters():
                param.requires_grad = True
        if model_name in self.model_projectors:
            for param in self.model_projectors[model_name].parameters():
                param.requires_grad = True
                
    def _create_target_model(self, model_name: str, device: str = "cuda"):
        """Create and return the target model for alignment using HuggingFace Hub only"""
        try:
            if model_name.startswith("clip_"):
                # Use HuggingFace CLIP model
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                return model.vision_model.to(device).eval()
                
            elif model_name.startswith("dinov2_"):
                # Use HuggingFace DINOv2 model
                model = AutoModel.from_pretrained("facebook/dinov2-base")
                return model.to(device).eval()
                
            elif model_name.startswith("dinov3_"):
                # Use HuggingFace DINOv3 model
                model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
                return model.to(device).eval()
                
            elif model_name.startswith("sam2_"):
                # Use HuggingFace SAM2 model
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
                model = predictor.model.image_encoder
                return model.to(device).eval()
                
            else:
                raise ValueError(f"Unsupported target model: {model_name}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load target model {model_name} from HuggingFace: {e}")
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def load_target_model_config(self, config_path: str = None):
        """Load target model configuration from file"""
        if config_path is None:
            config_path = "models/alignment_config.json"
            
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Please run prepare_alignment_models.py first")
            return {}
            
        with open(config_path) as f:
            return json.load(f)


class AlignmentLoss(nn.Module):
    """Loss functions for representation alignment."""
    
    def __init__(self, loss_type: str = 'mse', temperature: float = 0.1):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        
    def forward(self, cave_embeddings: torch.Tensor, 
                target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between CAVE and target embeddings.
        
        Args:
            cave_embeddings: CAVE model embeddings (B, num_tokens, embed_dim)
            target_embeddings: Target model embeddings (B, num_tokens, embed_dim)
            
        Returns:
            Alignment loss
        """
        if self.loss_type == 'mse':
            return F.mse_loss(cave_embeddings, target_embeddings)
        elif self.loss_type == 'cosine':
            # Normalize embeddings
            cave_norm = F.normalize(cave_embeddings, dim=-1)
            target_norm = F.normalize(target_embeddings, dim=-1)
            
            # Cosine similarity
            cos_sim = torch.sum(cave_norm * target_norm, dim=-1)
            
            # Convert to loss (1 - cosine_similarity)
            return 1.0 - cos_sim.mean()
        elif self.loss_type == 'contrastive':
            # Flatten to (B*num_tokens, embed_dim)
            cave_flat = cave_embeddings.view(-1, cave_embeddings.shape[-1])
            target_flat = target_embeddings.view(-1, target_embeddings.shape[-1])
            
            # Normalize
            cave_norm = F.normalize(cave_flat, dim=-1)
            target_norm = F.normalize(target_flat, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(cave_norm, target_norm.T) / self.temperature
            
            # Labels (diagonal should be positive pairs)
            labels = torch.arange(cave_flat.shape[0], device=cave_flat.device)
            
            # Contrastive loss
            loss = F.cross_entropy(sim_matrix, labels)
            return loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for CAVE encoder-decoder training.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize reconstruction loss.
        
        Args:
            loss_type: Type of reconstruction loss ('mse', 'l1', 'combined')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstructed: Reconstructed images (B, C, H, W)  
            original: Original images (B, C, H, W)
            
        Returns:
            Reconstruction loss scalar
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(reconstructed, original)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(reconstructed, original)
        elif self.loss_type == 'combined':
            mse_loss = F.mse_loss(reconstructed, original)
            l1_loss = F.l1_loss(reconstructed, original)
            loss = 0.8 * mse_loss + 0.2 * l1_loss
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {self.loss_type}")
            
        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint training of alignment and reconstruction tasks.
    """
    
    def __init__(self, task_weights: Dict[str, float] = None, 
                 alignment_loss_type: str = 'mse',
                 reconstruction_loss_type: str = 'mse',
                 alignment_temperature: float = 0.1,
                 target_models: Dict = None):
        """
        Initialize multi-task loss.
        
        Args:
            task_weights: Weights for different tasks (e.g., {'dinov2_base': 1.0, 'reconstruction': 1.0})
            alignment_loss_type: Type of alignment loss
            reconstruction_loss_type: Type of reconstruction loss
            alignment_temperature: Temperature for alignment loss
            target_models: Pre-loaded target models dict
        """
        super().__init__()
        
        # Store target models for use in loss computation
        self.target_models = target_models or {}
        
        # Default task weights
        if task_weights is None:
            task_weights = {
                'dinov2_base': 1.0,
                'clip_base': 1.0, 
                'sam2_base': 1.0,
                'reconstruction': 1.0
            }
        self.task_weights = task_weights
        
        # Initialize individual loss functions
        self.alignment_loss = AlignmentLoss(
            loss_type=alignment_loss_type,
            temperature=alignment_temperature
        )
        self.reconstruction_loss = ReconstructionLoss(
            loss_type=reconstruction_loss_type
        )
        
    def forward(self, task_outputs: Dict[str, Dict[str, torch.Tensor]], 
                target_embeddings: Dict[str, torch.Tensor] = None,
                original_images: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            task_outputs: Dictionary mapping task names to their outputs
            target_embeddings: Dictionary mapping alignment tasks to target embeddings
            original_images: Original images for reconstruction loss
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, task_output in task_outputs.items():
            if task_name == 'reconstruction':
                # Reconstruction loss
                if 'original_latent' in task_output:
                    # Use latent space reconstruction
                    reconstructed_latent = task_output['aligned_embedding']
                    original_latent = task_output['original_latent']
                    task_loss = self.reconstruction_loss(reconstructed_latent, original_latent)
                else:
                    # Use pixel space reconstruction (if original_images provided)
                    if original_images is not None:
                        reconstructed_images = task_output['aligned_embedding']
                        task_loss = self.reconstruction_loss(reconstructed_images, original_images)
                    else:
                        print(f"Warning: No target for reconstruction task")
                        task_loss = torch.tensor(0.0, device=task_output['aligned_embedding'].device)
            else:
                # Alignment loss
                if target_embeddings and task_name in target_embeddings:
                    predicted = task_output['aligned_embedding']
                    target = target_embeddings[task_name]
                    task_loss = self.alignment_loss(predicted, target)
                else:
                    print(f"Warning: No target embedding for alignment task {task_name}")
                    task_loss = torch.tensor(0.0, device=task_output['aligned_embedding'].device)
            
            # Apply task weight and accumulate
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_loss = task_weight * task_loss
            
            losses[f'loss/{task_name}'] = task_loss.detach()
            losses[f'weighted_loss/{task_name}'] = weighted_loss.detach()
            total_loss += weighted_loss
        
        losses['loss/total'] = total_loss
        
        return losses
