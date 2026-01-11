#!/bin/bash

# LLaVA v1.5 Fine-tuning with Dual Vision Tower
# This script demonstrates how to fine-tune LLaVA with the dual vision encoder

# ===== Configuration =====
# Dual Vision Tower Settings
VISION_TOWER="dual_vision_clip_openai/clip-vit-large-patch14-336"
DUAL_VISION_PRETRAINED="openai/clip-vit-large-patch14-336"
DUAL_VISION_OUTPUT_MODE="right"  # Options: right, left, concat, both
DUAL_VISION_TRAIN_RIGHT=False     # Set to True to train right branch, otherwise freeze all
DUAL_VISION_NUM_AUX_TOKENS=256   # Number of auxiliary tokens
DUAL_VISION_FLASH_ATTN=True
DUAL_VISION_ATTENTION_MODE="cross"  # Options: joint, cross

DUAL_VISION_ADAPTATION_CKPT="./dual_vision/checkpoints/best_model.pt"
PRETRAIN_MM_MLP_ADAPTER="./checkpoints/llava-v1.5-7b-dual-vision-pretrain/mm_projector.bin"

OUTPUT_DIR="./checkpoints/llava-v1.5-7b-dual-vision-finetune"

# ===== Training Command =====
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./models/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower ${VISION_TOWER} \
    --dual_vision_pretrained ${DUAL_VISION_PRETRAINED} \
    --dual_vision_output_mode ${DUAL_VISION_OUTPUT_MODE} \
    --dual_vision_train_right ${DUAL_VISION_TRAIN_RIGHT} \
    --dual_vision_num_aux_tokens ${DUAL_VISION_NUM_AUX_TOKENS} \
    --dual_vision_flash_attn ${DUAL_VISION_FLASH_ATTN} \
    --dual_vision_attention_mode ${DUAL_VISION_ATTENTION_MODE} \
    --dual_vision_adaptation_ckpt ${DUAL_VISION_ADAPTATION_CKPT} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
# 1. Adjust DATA_PATH, IMAGE_FOLDER, and PRETRAIN_MM_MLP_ADAPTER paths
# 2. This script freezes the left branch (dual_vision_freeze_left=True)
#    to only train the right branch and reduce memory usage
# 3. For full model fine-tuning, set:
#    --dual_vision_freeze_left False
# 4. The pretrained mm_projector should come from the pretraining stage
# 5. For different encoder types, change VISION_TOWER:
#    - SigLIP: dual_vision_siglip_google/siglip-so400m-patch14-384
#    - ViT: dual_vision_vit_google/vit-base-patch16-224

