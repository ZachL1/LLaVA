#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path checkpoints/llava-v1.5-13b-cave-pretrain \
    --model-base models/vicuna-13b-v1.5 \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    # --load-4bit