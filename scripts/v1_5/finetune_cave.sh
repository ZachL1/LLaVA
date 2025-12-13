#!/bin/bash

pip install transformers==4.37.2 accelerate==0.21.0
pip install deepspeed
pip install grpcio-status==1.33.2 protobuf==3.19.6
pip install omegaconf
pip install -U wandb
wandb login --relogin xxx
# pip install hydra-core mup diffusers==0.18.0 huggingface-hub==0.19.3
# models/vicuna-13b-v1.5/generation_config.json add "do_sample": true

for token in 1 2 64 256
do
    VISION_TOWER=cave # cave without kl, without learnable
    # VISION_TOWER=cave-kl # cave with kl, without learnable
    # VISION_TOWER=cave-learnable # cave without kl, with learnable
    # VISION_TOWER=cave-kl-learnable # cave with kl, with learnable
    CAVE_CKPT=/mnt/bn/kgvanasgcparnold/annan/CAVE/cave_flexible/checkpoint-500000/converted_weights/

    deepspeed --num_gpus=8 --master_port 9965 llava/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path ./models/vicuna-13b-v1.5 \
        --version v1 \
        --data_path ./playground/data/llava_v1_5_mix665k.json \
        --image_folder ./playground/data \
        --output_dir ./checkpoints/llava-v1.5-13b-$VISION_TOWER-finetune-$token \
        --vision_tower $VISION_TOWER-$token \
        --cave_config ./cave/config.yaml \
        --cave_ckpt $CAVE_CKPT \
        --cave_token $token \
        --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-$VISION_TOWER-pretrain-$token/mm_projector.bin \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
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
done
