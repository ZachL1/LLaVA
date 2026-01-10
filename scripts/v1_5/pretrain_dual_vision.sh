#!/bin/bash

# LLaVA v1.5 Pretraining with Dual Vision Tower
# This script demonstrates how to pretrain LLaVA with the dual vision encoder

# ===== Configuration =====
# Dual Vision Tower Settings
VISION_TOWER="dual_vision_clip_openai/clip-vit-large-patch14-336"
DUAL_VISION_PRETRAINED="openai/clip-vit-large-patch14-336"
DUAL_VISION_OUTPUT_MODE="right"  # Options: right, left, concat, both
DUAL_VISION_TRAIN_RIGHT=False    # Set to True to train right branch, otherwise freeze all
DUAL_VISION_NUM_AUX_TOKENS=256   # Number of auxiliary tokens (None for auto)
DUAL_VISION_FLASH_ATTN=True
DUAL_VISION_ATTENTION_MODE="cross"  # Options: joint, cross
# NEW: Path to adaptation trained checkpoint
# This checkpoint contains the right branch (MoT) weights aligned with teacher
DUAL_VISION_ADAPTATION_CKPT="./dual_vision/checkpoints/best_model.pt"

OUTPUT_DIR="./checkpoints/llava-v1.5-7b-dual-vision-pretrain"

# ===== Training Command =====
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./models/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower ${VISION_TOWER} \
    --dual_vision_pretrained ${DUAL_VISION_PRETRAINED} \
    --dual_vision_output_mode ${DUAL_VISION_OUTPUT_MODE} \
    --dual_vision_train_right ${DUAL_VISION_TRAIN_RIGHT} \
    --dual_vision_num_aux_tokens ${DUAL_VISION_NUM_AUX_TOKENS} \
    --dual_vision_flash_attn ${DUAL_VISION_FLASH_ATTN} \
    --dual_vision_attention_mode ${DUAL_VISION_ATTENTION_MODE} \
    --dual_vision_adaptation_ckpt ${DUAL_VISION_ADAPTATION_CKPT} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# ===== Notes =====
# 1. Adjust DATA_PATH and IMAGE_FOLDER to your dataset paths
# 2. For SigLIP base model, use:
#    --vision_tower dual_vision_siglip_google/siglip-so400m-patch14-384
# 3. For ViT base model, use:
#    --vision_tower dual_vision_vit_google/vit-base-patch16-224
# 4. If using concat output mode, hidden size will be 2x:
#    --dual_vision_output_mode concat
# 5. To train right branch, set:
#    --dual_vision_train_right True

