

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  train_adaptation.py \
  --encoder_type clip \
  --pretrained_model openai/clip-vit-large-patch14-336 \
  --data_path /test/annan/ImageNet-1k/train \
  --val_data_path /test/annan/ImageNet-1k/train_val \
  --num_epochs 10 \
  --batch_size 16