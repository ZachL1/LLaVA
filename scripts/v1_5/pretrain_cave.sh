#!/bin/bash

VISION_TOWER=cave-pretrain-256 # cave without kl, without learnable
CAVE_CKPT=/mnt/bn/kgvanasgcparnold/annan/CAVE/cave_flexible/checkpoint-500000/converted_weights/
# VISION_TOWER=cave-kl-pretrain # cave with kl, without learnable
# VISION_TOWER=cave-learnable-pretrain # cave without kl, with learnable
# VISION_TOWER=cave-kl-learnable-pretrain # cave with kl, with learnable
# CAVE_CKPT=/mnt/bn/kgvanasgcparnold/annan/DiT/results/checkpoint-400000/converted_weights/

pip install deepspeed
pip install grpcio-status==1.33.2 protobuf==3.19.6
pip install omegaconf

# for flextoken:
pip install hydra-core mup diffusers==0.18.0 huggingface-hub==0.19.3

pip install -U wandb
wandb login --relogin 2943b45498fccaa5f1941a06bcc942bf3bea49fb

deepspeed --num_gpus=8 --master_port 9965 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./models/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --output_dir ./checkpoints/llava-v1.5-13b-$VISION_TOWER \
    --vision_tower $VISION_TOWER \
    --cave_config ./cave/config.yaml \
    --cave_ckpt $CAVE_CKPT \
    --cave_token 256 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
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
