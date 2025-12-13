
# pip install transformers==4.37.2 accelerate==0.21.0
# pip install deepspeed
# pip install grpcio-status==1.33.2 protobuf==3.19.6
# pip install omegaconf
# pip install -U wandb
# pip install hydra-core mup diffusers==0.18.0 huggingface-hub==0.19.3

# for token in 1 256 2 4 8 16 32 64 128; do

#     # ckpt=/mnt/bn/kgvanasgcparnold/annan/LLaVA/checkpoints/llava-v1.5-13b-flex-finetune-1
#     ckpt=/mnt/bn/kgvanasgcparnold/annan/LLaVA/checkpoints/llava-v1.5-13b-cave-finetune-$token

#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh $ckpt > logs/gqa-$token.log &

#     CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh $ckpt > logs/sqa-$token.log &

#     CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/eval/textvqa.sh $ckpt > logs/textvqa-$token.log &



#     CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/pope.sh $ckpt > logs/pope-$token.log &

#     # pip install scikit-learn
#     CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval/mme.sh $ckpt > logs/mme-$token.log &

#     # pip install openai
#     rm -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
#     CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/eval/llavabench.sh $ckpt > logs/llavabench-$token.log &

#     wait
# done

# # for token in 16 64; do
# for token in 1; do

# ckpt=/mnt/bn/kgvanasgcparnold/annan/LLaVA/checkpoints/llava-v1.5-13b-cave-finetune-$token

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh $ckpt

# # CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vizwiz.sh $ckpt

# done




# for token in 1 32 128 256; do
for token in 1; do

ckpt=/mnt/bn/kgvanasgcparnold/annan/LLaVA/checkpoints/llava-v1.5-13b-flex-finetune-$token

CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh $ckpt

done






