"""CLIP Vision Encoder based Dual Vision Encoder"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

try:
    from .config import DualVisionConfig
    from .dual_vision_encoder import DualVisionEncoder
except ImportError:
    from config import DualVisionConfig
    from dual_vision_encoder import DualVisionEncoder


class DualCLIPVisionEncoder(DualVisionEncoder):
    """Dual Vision Encoder based on CLIP Vision Encoder architecture.
    
    Supports loading weights from:
    - HuggingFace transformers CLIP models
    - OpenAI CLIP checkpoints
    """
    
    def __init__(self, config: DualVisionConfig):
        # Ensure CLIP-specific settings
        if not config.use_cls_token:
            warnings.warn("CLIP uses CLS token, setting use_cls_token=True")
            config.use_cls_token = True
        
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Convenience method to create encoder from pretrained CLIP model."""
        # Load config from pretrained model
        try:
            from transformers import CLIPVisionConfig, CLIPVisionModel
            
            clip_model = CLIPVisionModel.from_pretrained(model_name)
            clip_config = clip_model.config
            
            # Create our config from CLIP config
            config = DualVisionConfig(
                encoder_type='clip',
                hidden_size=clip_config.hidden_size,
                num_layers=clip_config.num_hidden_layers,
                num_heads=clip_config.num_attention_heads,
                intermediate_size=clip_config.intermediate_size,
                image_size=clip_config.image_size,
                patch_size=clip_config.patch_size,
                num_channels=clip_config.num_channels,
                layer_norm_eps=clip_config.layer_norm_eps,
                **kwargs
            )
            
            # Create encoder
            encoder = cls(config)
            
            # Load weights
            encoder.load_pretrained_weights(model_name)
            
            return encoder
        except ImportError:
            raise ImportError("transformers library is required to use from_pretrained")
    
    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        strict: bool = False
    ):
        """Load pretrained CLIP vision encoder weights.
        
        Args:
            checkpoint_path: HuggingFace model name or path
            strict: Whether to strictly enforce key matching
        """
        print(f"Loading pretrained CLIP weights from: {checkpoint_path}")
        
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            
            # Load CLIP model
            clip_model = CLIPVisionModel.from_pretrained(checkpoint_path)
            state_dict = clip_model.state_dict()
            
            # Convert weights
            converted = self._convert_clip_weights(state_dict)
            
            # Load dual state dict
            self._load_dual_state_dict(converted, strict=strict)
            
            # Set preprocessor from pretrained model
            try:
                self._preprocessor = CLIPImageProcessor.from_pretrained(checkpoint_path)
                print(f"Loaded preprocessor from {checkpoint_path}")
            except:
                print("Could not load preprocessor, using default")
            
            print("Successfully loaded CLIP vision encoder weights")
            
        except Exception as e:
            raise ValueError(f"Could not load CLIP weights from {checkpoint_path}: {e}")
    
    def _convert_clip_weights(self, state_dict: dict) -> dict:
        """Convert CLIP vision encoder weights to our format."""
        converted = {}
        
        for key, value in state_dict.items():
            # Remove 'vision_model.' prefix
            if key.startswith('vision_model.'):
                key = key[13:]
            
            # Map embeddings
            if key == 'embeddings.class_embedding':
                converted['cls_token'] = value.unsqueeze(0).unsqueeze(0)  # [hidden_size] -> [1, 1, hidden_size]
            elif key == 'embeddings.position_embedding.weight':
                converted['position_embeddings'] = value.unsqueeze(0)  # [num_pos, hidden_size] -> [1, num_pos, hidden_size]
            elif key.startswith('embeddings.patch_embedding'):
                new_key = key.replace('embeddings.patch_embedding', 'patch_embedding.projection')
                converted[new_key] = value
            
            # Map encoder layers
            elif key.startswith('encoder.layers.'):
                parts = key.split('.')
                layer_idx = parts[2]
                rest = '.'.join(parts[3:])
                
                # Attention
                if 'self_attn.q_proj' in rest:
                    new_key = f'layers.{layer_idx}.attention.q_proj' + rest.split('q_proj')[1]
                    converted[new_key] = value
                elif 'self_attn.k_proj' in rest:
                    new_key = f'layers.{layer_idx}.attention.k_proj' + rest.split('k_proj')[1]
                    converted[new_key] = value
                elif 'self_attn.v_proj' in rest:
                    new_key = f'layers.{layer_idx}.attention.v_proj' + rest.split('v_proj')[1]
                    converted[new_key] = value
                elif 'self_attn.out_proj' in rest:
                    new_key = f'layers.{layer_idx}.attention.o_proj' + rest.split('out_proj')[1]
                    converted[new_key] = value
                
                # Layer norms
                elif 'layer_norm1' in rest:
                    new_key = f'layers.{layer_idx}.input_layernorm' + rest.split('layer_norm1')[1]
                    converted[new_key] = value
                elif 'layer_norm2' in rest:
                    new_key = f'layers.{layer_idx}.post_attention_layernorm' + rest.split('layer_norm2')[1]
                    converted[new_key] = value
                
                # MLP
                elif 'mlp.fc1' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc1' + rest.split('fc1')[1]
                    converted[new_key] = value
                elif 'mlp.fc2' in rest:
                    new_key = f'layers.{layer_idx}.mlp.fc2' + rest.split('fc2')[1]
                    converted[new_key] = value
            
            # Pre or post layernorm
            elif key.startswith('pre_layrnorm') or key.startswith('pre_layernorm'):
                # CLIP uses pre-layernorm, we'll skip this as it's handled per-layer
                pass
            elif key.startswith('post_layernorm'):
                converted['final_layernorm' + key[14:]] = value
        
        return converted
    
    def _load_dual_state_dict(self, state_dict: dict, strict: bool = False):
        """Load state dict for both left and right branches."""
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
            should_duplicate = any(mot_key in key for mot_key in mot_keys)
            
            if should_duplicate:
                # Special handling for different module types
                # For MLP: layers.X.mlp.fc1.weight -> layers.X.mlp_mot.fc1.weight
                # For others: layers.X.attention.q_proj.weight -> layers.X.attention.q_proj_mot.weight
                if '.mlp.' in key:
                    # Replace mlp with mlp_mot
                    mot_key = key.replace('.mlp.', '.mlp_mot.')
                elif any(param_name in key for param_name in ['input_layernorm', 'post_attention_layernorm']):
                    # Replace layernorm with layernorm_mot
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
            # print(f"Missing keys: {missing[:10]}..." if len(missing) > 10 else f"Missing keys: {missing}")
            print("Missing keys:", missing)
        if unexpected:
            # print(f"Unexpected keys: {unexpected[:10]}..." if len(unexpected) > 10 else f"Unexpected keys: {unexpected}")
            print("Unexpected keys:", unexpected)
        
        print(f"Loaded {len(dual_state_dict)} keys total")
        print(f"Left branch: {len([k for k in dual_state_dict.keys() if '_mot' not in k])} params")
        print(f"Right branch: {len([k for k in dual_state_dict.keys() if '_mot' in k])} params")
