#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        
        # Handle CAVE configuration
        if "cave" in vision_tower:
            self.config.cave_config = model_args.cave_config
            self.config.cave_ckpt = model_args.cave_ckpt
            self.config.cave_token = model_args.cave_token
        
        # Handle dual_vision configuration
        if "dual_vision" in vision_tower:
            self.config.dual_vision_pretrained = getattr(model_args, 'dual_vision_pretrained', None)
            self.config.dual_vision_num_aux_tokens = getattr(model_args, 'dual_vision_num_aux_tokens', None)
            self.config.dual_vision_output_mode = getattr(model_args, 'dual_vision_output_mode', 'right')
            self.config.dual_vision_train_right = getattr(model_args, 'dual_vision_train_right', False)
            self.config.dual_vision_flash_attn = getattr(model_args, 'dual_vision_flash_attn', True)
            self.config.dual_vision_attention_mode = getattr(model_args, 'dual_vision_attention_mode', 'joint')
            self.config.dual_vision_adaptation_ckpt = getattr(model_args, 'dual_vision_adaptation_ckpt', None)

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        
        # Initialize LLM hidden projection for DualVisionTower
        if "dual_vision" in model_args.vision_tower and hasattr(vision_tower, 'set_llm_hidden_size'):
            # Get LLM hidden size from config
            llm_hidden_size = self.config.hidden_size
            vision_tower.set_llm_hidden_size(llm_hidden_size)
            print(f"Initialized DualVisionTower with LLM hidden size: {llm_hidden_size}")

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _get_text_hidden_states(self, input_ids, attention_mask=None, labels=None):
        """Get LLM hidden states for non-target text tokens only (to avoid information leakage).
        
        IMPORTANT: To avoid information leakage during training, we only use
        text tokens that are NOT part of the training target. This is determined by:
        
        1. If labels are provided: Use tokens where labels == IGNORE_INDEX
           (these are masked tokens that the model doesn't predict)
        2. If no labels: Fall back to using only tokens BEFORE the first IMAGE_TOKEN
           (conservative approach that ensures safety)
        
        Training data format (v1): 
            [系统提示] USER: <image>\n[问题] ASSISTANT: [回复]
            ↑----- IGNORE_INDEX -----↑        ↑-- TARGET --↑
        
        We use tokens where labels == IGNORE_INDEX, excluding IMAGE_TOKEN itself.
        This includes system prompt + user question, but NOT the assistant response.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Optional attention mask [batch_size, seq_length]
            labels: Optional labels tensor [batch_size, seq_length]. 
                    Tokens with label == IGNORE_INDEX are safe to use.
            
        Returns:
            text_hidden_states: List of hidden states for each batch item,
                                each of shape [1, text_seq_len, hidden_dim]
        """
        batch_size = input_ids.shape[0]
        text_hidden_states_list = []
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            
            if labels is not None:
                # Use labels to determine which tokens are safe (not training targets)
                cur_labels = labels[batch_idx]
                
                # Safe tokens: labels == IGNORE_INDEX (not a prediction target)
                # Also exclude IMAGE_TOKEN and padding
                safe_mask = (cur_labels == IGNORE_INDEX) & (cur_input_ids != IMAGE_TOKEN_INDEX)
                
                if attention_mask is not None:
                    safe_mask = safe_mask & attention_mask[batch_idx].bool()
                
                text_token_ids = cur_input_ids[safe_mask]
            else:
                # Fallback: use only tokens BEFORE the first IMAGE_TOKEN (conservative)
                image_token_positions = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                
                if len(image_token_positions) == 0:
                    # No image token found, use all non-padding tokens
                    if attention_mask is not None:
                        valid_mask = attention_mask[batch_idx].bool()
                        text_token_ids = cur_input_ids[valid_mask]
                    else:
                        text_token_ids = cur_input_ids
                else:
                    # Get the first image token position
                    first_image_pos = image_token_positions[0].item()
                    
                    # Only use tokens BEFORE the image token
                    text_token_ids = cur_input_ids[:first_image_pos]
                    
                    # Filter out padding tokens if attention_mask is provided
                    if attention_mask is not None and first_image_pos > 0:
                        cur_attention_mask = attention_mask[batch_idx][:first_image_pos]
                        valid_mask = cur_attention_mask.bool()
                        text_token_ids = text_token_ids[valid_mask]
            
            if len(text_token_ids) == 0:
                # No safe text tokens, create a dummy hidden state
                hidden_dim = self.config.hidden_size
                dummy_hs = torch.zeros(1, 1, hidden_dim, device=input_ids.device, dtype=self.dtype)
                text_hidden_states_list.append(dummy_hs)
                continue
            
            # Get embeddings for safe text tokens
            text_embeds = self.get_model().embed_tokens(text_token_ids.unsqueeze(0))  # [1, text_len, hidden_dim]
            
            # Run through the LLM to get hidden states
            # We use the base model forward with output_hidden_states=True
            with torch.no_grad():
                outputs = self.get_model()(
                    inputs_embeds=text_embeds,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            
            # Get the last hidden state (output of all transformer layers)
            # outputs.hidden_states is a tuple of (embedding_output, layer1_output, ..., layerN_output)
            # We take the last one which is the output of the final layer
            last_hidden_state = outputs.hidden_states[-1]  # [1, text_len, hidden_dim]
            
            text_hidden_states_list.append(last_hidden_state)
        
        return text_hidden_states_list

    def encode_images(self, images, text_hidden_states=None):
        """Encode images through the vision tower.
        
        Args:
            images: Image tensor or list of image tensors
            text_hidden_states: Optional list of LLM hidden states for dual vision tower
            
        Returns:
            Image features after projection
        """
        vision_tower = self.get_model().get_vision_tower()
        
        # Check if this is a DualVisionTower that can use text_hidden_states
        if hasattr(vision_tower, 'llm_hidden_proj') and text_hidden_states is not None:
            image_features = vision_tower(images, text_hidden_states=text_hidden_states)
        else:
            image_features = vision_tower(images)
        
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Get text hidden states for DualVisionTower if applicable
        # IMPORTANT: Pass labels to avoid information leakage (only use tokens where labels == IGNORE_INDEX)
        text_hidden_states = None
        if hasattr(vision_tower, 'llm_hidden_proj') and vision_tower.llm_hidden_proj is not None:
            # print("Getting text hidden states for DualVisionTower")
            text_hidden_states = self._get_text_hidden_states(input_ids, attention_mask, labels=labels)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            
            # For list/multi-image case, we need to handle text_hidden_states appropriately
            # Each image in the batch should get corresponding text hidden states
            if text_hidden_states is not None:
                # Expand text_hidden_states to match concat_images
                # If we have multiple images per sample, replicate the text hidden states
                expanded_text_hs = []
                split_sizes = [image.shape[0] for image in images]
                for batch_idx, num_imgs in enumerate(split_sizes):
                    for _ in range(num_imgs):
                        expanded_text_hs.append(text_hidden_states[batch_idx])
                image_features = self.encode_images(concat_images, text_hidden_states=expanded_text_hs)
            else:
                image_features = self.encode_images(concat_images)
            
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # Single batch of images
            if text_hidden_states is not None:
                # Stack text hidden states into a batch tensor
                # Each element is [1, seq_len, hidden_dim], but seq_lens may differ
                # For batch processing, we need to handle variable lengths
                max_seq_len = max(ths.shape[1] for ths in text_hidden_states)
                hidden_dim = text_hidden_states[0].shape[2]
                batch_size = len(text_hidden_states)
                
                # Pad to uniform length and stack
                padded_hs = torch.zeros(batch_size, max_seq_len, hidden_dim, 
                                       device=text_hidden_states[0].device,
                                       dtype=text_hidden_states[0].dtype)
                for i, ths in enumerate(text_hidden_states):
                    seq_len = ths.shape[1]
                    padded_hs[i, :seq_len, :] = ths[0]
                
                image_features = self.encode_images(images, text_hidden_states=padded_hs)
            else:
                image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
