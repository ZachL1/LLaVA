#! /bin/bash

# Pretrain Data
python download.py


# Visual Instruction Tuning Data

mkdir coco && cd coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
cd ..

mkdir gqa && cd gqa
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
cd ..

mkdir textvqa && cd textvqa
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
cd ..

mkdir vg && cd vg
mkdir VG_100K && cd VG_100K
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip
cd ..

mkdir VG_100K_2 && cd VG_100K_2
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip
cd ..

cd ..

cd ocr_vqa
# https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_
python loadDataset.py
cd ..