"""SigLIP Vision Encoder based Dual Vision Encoder"""

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


class DualSigLIPVisionEncoder(DualVisionEncoder):
    """Dual Vision Encoder based on SigLIP Vision Encoder architecture.
    
    SigLIP differences from ViT/CLIP:
    - No CLS token (uses mean pooling)
    - No bias in attention layers
    - Different normalization scheme
    """
    
    def __init__(self, config: DualVisionConfig):
        # Ensure SigLIP-specific settings
        if config.use_cls_token:
            warnings.warn("SigLIP doesn't use CLS token, setting use_cls_token=False")
            config.use_cls_token = False
        
        if config.pooling_type != 'mean':
            warnings.warn("SigLIP uses mean pooling, setting pooling_type='mean'")
            config.pooling_type = 'mean'
        
        if config.use_bias_in_attention:
            warnings.warn("SigLIP doesn't use bias in attention, setting use_bias_in_attention=False")
            config.use_bias_in_attention = False
        
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Convenience method to create encoder from pretrained SigLIP model."""
        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
            
            siglip_model = SiglipVisionModel.from_pretrained(model_name)
            siglip_config = siglip_model.config
            
            # Create our config from SigLIP config
            config = DualVisionConfig(
                encoder_type='siglip',
                hidden_size=siglip_config.hidden_size,
                num_layers=siglip_config.num_hidden_layers,
                num_heads=siglip_config.num_attention_heads,
                intermediate_size=siglip_config.intermediate_size,
                image_size=siglip_config.image_size,
                patch_size=siglip_config.patch_size,
                num_channels=siglip_config.num_channels,
                layer_norm_eps=siglip_config.layer_norm_eps,
                use_cls_token=False,
                pooling_type='mean',
                use_bias_in_attention=False,
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
        """Load pretrained SigLIP vision encoder weights.
        
        Args:
            checkpoint_path: HuggingFace model name or path
            strict: Whether to strictly enforce key matching
        """
        print(f"Loading pretrained SigLIP weights from: {checkpoint_path}")
        
        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            
            # Load SigLIP model
            siglip_model = SiglipVisionModel.from_pretrained(checkpoint_path)
            state_dict = siglip_model.state_dict()
            
            # Convert weights
            converted = self._convert_siglip_weights(state_dict)
            
            # Load dual state dict
            self._load_dual_state_dict(converted, strict=strict)
            
            # Set preprocessor from pretrained model
            try:
                self._preprocessor = SiglipImageProcessor.from_pretrained(checkpoint_path)
                print(f"Loaded preprocessor from {checkpoint_path}")
            except:
                print("Could not load preprocessor, using default")
            
            print("Successfully loaded SigLIP vision encoder weights")
            
        except Exception as e:
            raise ValueError(f"Could not load SigLIP weights from {checkpoint_path}: {e}")
    
    def _convert_siglip_weights(self, state_dict: dict) -> dict:
        """Convert SigLIP vision encoder weights to our format."""
        converted = {}
        
        for key, value in state_dict.items():
            # Remove 'vision_model.' prefix
            if key.startswith('vision_model.'):
                key = key[13:]
            
            # Map embeddings (no CLS token in SigLIP)
            if key == 'embeddings.position_embedding.weight':
                converted['position_embeddings'] = value.unsqueeze(0)
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
            
            # Post layernorm
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
        # Note: No cls_token for SigLIP
        mot_keys = [
            'attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.o_proj',
            'input_layernorm', 'post_attention_layernorm',
            'mlp.fc1', 'mlp.fc2',
            'final_layernorm'
        ]
        
        for key, value in state_dict.items():
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
        
        print(f"Loaded {len(dual_state_dict)} keys total")
        print(f"Left branch: {len([k for k in dual_state_dict.keys() if '_mot' not in k])} params")
        print(f"Right branch: {len([k for k in dual_state_dict.keys() if '_mot' in k])} params")
