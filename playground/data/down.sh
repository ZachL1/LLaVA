#! /bin/bash

# Pretrain Data
python download.py


# Visual Instruction Tuning Data
cd Visual-Instruction-Tuning
wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json

mkdir coco && cd coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
cd ..

mkdir -p gqa && cd gqa
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip -o images.zip &
cd ..

mkdir -p textvqa && cd textvqa
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip -o train_val_images.zip &
cd ..

mkdir -p vg && cd vg
mkdir -p VG_100K && cd VG_100K
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip -o images.zip &
cd ..

mkdir -p VG_100K_2 && cd VG_100K_2
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -o images2.zip &
cd ..

cd ..

cd ocr_vqa
# https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_
python loadDataset.py
cd ..