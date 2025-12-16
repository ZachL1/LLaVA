"""ViT-based Dual Vision Encoder"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

try:
    from .config import DualVisionConfig
    from .dual_vision_encoder import DualVisionEncoder
    from .utils import rename_state_dict_keys
except ImportError:
    from config import DualVisionConfig
    from dual_vision_encoder import DualVisionEncoder
    from utils import rename_state_dict_keys


class DualViTEncoder(DualVisionEncoder):
    """Dual Vision Encoder based on Vision Transformer (ViT) architecture.
    
    Supports loading weights from:
    - HuggingFace transformers ViT models
    - timm ViT models
    - Custom ViT checkpoints
    """
    
    def __init__(self, config: DualVisionConfig):
        # Ensure ViT-specific settings
        if not config.use_cls_token:
            warnings.warn("ViT typically uses CLS token, setting use_cls_token=True")
            config.use_cls_token = True
        
        super().__init__(config)
    
    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False,
        library: str = 'auto'
    ):
        """Load pretrained ViT weights and duplicate for MoT branch.
        
        Args:
            checkpoint_path: Path or HuggingFace model name
            strict: Whether to strictly enforce key matching
            library: Which library format ('auto', 'transformers', 'timm')
        """
        print(f"Loading pretrained ViT weights from: {checkpoint_path}")
        
        # Try to load from HuggingFace transformers
        if library in ['auto', 'transformers']:
            try:
                from transformers import ViTModel, ViTImageProcessor
                pretrained_model = ViTModel.from_pretrained(checkpoint_path)
                state_dict = pretrained_model.state_dict()
                state_dict = self._convert_transformers_vit_weights(state_dict)
                self._load_dual_state_dict(state_dict, strict=strict)
                
                # Set preprocessor from pretrained model
                try:
                    self._preprocessor = ViTImageProcessor.from_pretrained(checkpoint_path)
                    print(f"Loaded preprocessor from {checkpoint_path}")
                except:
                    print("Could not load preprocessor, using default")
                
                print("Successfully loaded from HuggingFace transformers")
                return
            except Exception as e:
                if library == 'transformers':
                    raise e
                print(f"Failed to load from transformers: {e}")
        
        # Try to load from timm
        if library in ['auto', 'timm']:
            try:
                import timm
                pretrained_model = timm.create_model(checkpoint_path, pretrained=True)
                state_dict = pretrained_model.state_dict()
                state_dict = self._convert_timm_vit_weights(state_dict)
                self._load_dual_state_dict(state_dict, strict=strict)
                print("Successfully loaded from timm")
                return
            except Exception as e:
                if library == 'timm':
                    raise e
                print(f"Failed to load from timm: {e}")
        
        # Try to load as raw checkpoint
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            # Auto-detect format and convert
            if 'vision_model.encoder' in str(list(state_dict.keys())):
                state_dict = self._convert_transformers_vit_weights(state_dict)
            self._load_dual_state_dict(state_dict, strict=strict)
            print("Successfully loaded from checkpoint file")
        except Exception as e:
            raise ValueError(f"Could not load checkpoint from {checkpoint_path}: {e}")
    
    def _convert_transformers_vit_weights(self, state_dict: dict) -> dict:
        """Convert HuggingFace transformers ViT weights to our format."""
        converted = {}
        
        for key, value in state_dict.items():
            # Remove 'vit.' or 'vision_model.' prefix if present
            if key.startswith('vit.'):
                key = key[4:]
            elif key.startswith('vision_model.'):
                key = key[13:]
            
            # Map embeddings
            if key == 'embeddings.cls_token':
                converted['cls_token'] = value
            elif key == 'embeddings.position_embeddings':
                converted['position_embeddings'] = value
            elif key.startswith('embeddings.patch_embeddings.projection'):
                new_key = key.replace('embeddings.patch_embeddings.projection', 'patch_embedding.projection')
                converted[new_key] = value
            
            # Map encoder layers
            elif key.startswith('encoder.layer.'):
                # encoder.layer.0.attention.attention.query.weight -> layers.0.attention.q_proj.weight
                parts = key.split('.')
                layer_idx = parts[2]
                rest = '.'.join(parts[3:])
                
                if 'attention.attention.query' in rest:
                    new_key = f'layers.{layer_idx}.attention.q_proj' + rest.split('query')[1]
                    converted[new_key] = value
                elif 'attention.attention.key' in rest:
                    new_key = f'layers.{layer_idx}.attention.k_proj' + rest.split('key')[1]
                    converted[new_key] = value
                elif 'attention.attention.value' in rest:
                    new_key = f'layers.{layer_idx}.attention.v_proj' + rest.split('value')[1]
                    converted[new_key] = value
                elif 'attention.output.dense' in rest:
                    new_key = f'layers.{layer_idx}.attention.o_proj' + rest.split('dense')[1]
                    converted[new_key] = value
                elif 'layernorm_before' in rest:
                    new_key = f'layers.{layer_idx}.input_layernorm' + rest.split('layernorm_before')[1]
                    converted[new_key] = value
                elif 'layernorm_after' in rest:
                    new_key = f'layers.{layer_idx}.post_attention_layernorm' + rest.split('layernorm_after')[1]
                    converted[new_key] = value
                elif 'intermediate.dense' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc1' + rest.split('dense')[1]
                    converted[new_key] = value
                elif 'output.dense' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc2' + rest.split('dense')[1]
                    converted[new_key] = value
            
            # Final layernorm
            elif key.startswith('layernorm'):
                converted['final_layernorm' + key[9:]] = value
        
        return converted
    
    def _convert_timm_vit_weights(self, state_dict: dict) -> dict:
        """Convert timm ViT weights to our format."""
        converted = {}
        
        for key, value in state_dict.items():
            # Map embeddings
            if key == 'cls_token':
                converted['cls_token'] = value
            elif key == 'pos_embed':
                # timm includes CLS token in pos_embed, we separate it
                converted['position_embeddings'] = value
            elif key.startswith('patch_embed.proj'):
                new_key = key.replace('patch_embed.proj', 'patch_embedding.projection')
                converted[new_key] = value
            
            # Map blocks (transformer layers)
            elif key.startswith('blocks.'):
                parts = key.split('.')
                layer_idx = parts[1]
                rest = '.'.join(parts[2:])
                
                if 'attn.qkv' in rest:
                    # timm uses fused qkv, we need to split it
                    # We'll handle this specially after loading
                    converted[f'layers.{layer_idx}._fused_qkv' + rest.split('qkv')[1]] = value
                elif 'attn.proj' in rest:
                    new_key = f'layers.{layer_idx}.attention.o_proj' + rest.split('proj')[1]
                    converted[new_key] = value
                elif 'norm1' in rest:
                    new_key = f'layers.{layer_idx}.input_layernorm' + rest.split('norm1')[1]
                    converted[new_key] = value
                elif 'norm2' in rest:
                    new_key = f'layers.{layer_idx}.post_attention_layernorm' + rest.split('norm2')[1]
                    converted[new_key] = value
                elif 'mlp.fc1' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc1' + rest.split('fc1')[1]
                    converted[new_key] = value
                elif 'mlp.fc2' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc2' + rest.split('fc2')[1]
                    converted[new_key] = value
            
            # Final norm
            elif key.startswith('norm'):
                converted['final_layernorm' + key[4:]] = value
        
        # Handle fused qkv weights
        fused_keys = [k for k in converted.keys() if '_fused_qkv' in k]
        for fused_key in fused_keys:
            layer_idx = fused_key.split('.')[1]
            suffix = fused_key.split('_fused_qkv')[1]
            qkv_weight = converted[fused_key]
            
            # Split into q, k, v
            hidden_size = qkv_weight.shape[1] if len(qkv_weight.shape) > 1 else qkv_weight.shape[0]
            chunk_size = qkv_weight.shape[0] // 3
            
            q, k, v = qkv_weight.chunk(3, dim=0)
            converted[f'layers.{layer_idx}.attention.q_proj{suffix}'] = q
            converted[f'layers.{layer_idx}.attention.k_proj{suffix}'] = k
            converted[f'layers.{layer_idx}.attention.v_proj{suffix}'] = v
            
            # Remove fused key
            del converted[fused_key]
        
        return converted
    
    def _load_dual_state_dict(self, state_dict: dict, strict: bool = False):
        """Load state dict for both left and right branches.
        
        Args:
            state_dict: Converted state dict for left branch
            strict: Whether to use strict loading
        """
        # Create dual state dict
        dual_state_dict = {}
        
        # Left branch: use original keys
        for key, value in state_dict.items():
            dual_state_dict[key] = value
        
        # Right branch: duplicate with _mot suffix
        # Note: position_embedding_mot is NOT copied (it's a 1D Embedding, learned separately)
        mot_keys = [
            'cls_token',  # Right branch shares CLS token concept
            'attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj',
            'input_layernorm', 'post_attention_layernorm',
            'mlp.fc1', 'mlp.fc2',
            'final_layernorm'
        ]
        
        for key, value in state_dict.items():
            # Check if this key should be duplicated for MoT
            should_duplicate = any(mot_key in key for mot_key in mot_keys)
            
            if should_duplicate:
                # Special handling for different module types
                if '.mlp.' in key:
                    mot_key = key.replace('.mlp.', '.mlp_mot.')
                elif any(param_name in key for param_name in ['input_layernorm', 'post_attention_layernorm']):
                    if 'input_layernorm' in key:
                        mot_key = key.replace('input_layernorm', 'input_layernorm_mot')
                    else:
                        mot_key = key.replace('post_attention_layernorm', 'post_attention_layernorm_mot')
                elif 'final_layernorm' in key:
                    mot_key = key.replace('final_layernorm', 'final_layernorm_mot')
                elif 'cls_token' in key:
                    mot_key = key.replace('cls_token', 'cls_token_mot')
                else:
                    # For attention projections: add _mot before the final component
                    parts = key.rsplit('.', 1)
                    if len(parts) == 2:
                        mot_key = parts[0] + '_mot.' + parts[1]
                    else:
                        mot_key = key + '_mot'
                
                dual_state_dict[mot_key] = value.clone()
        
        # Load into model
        missing, unexpected = self.load_state_dict(dual_state_dict, strict=False)
        
        if missing:
            print(f"Missing keys: {missing[:10]}..." if len(missing) > 10 else f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:10]}..." if len(unexpected) > 10 else f"Unexpected keys: {unexpected}")
        
        print(f"Loaded pretrained weights with {len(dual_state_dict)} keys")
        print(f"Left branch parameters loaded: {len([k for k in dual_state_dict.keys() if '_mot' not in k])}")
        print(f"Right branch (_mot) parameters loaded: {len([k for k in dual_state_dict.keys() if '_mot' in k])}")
