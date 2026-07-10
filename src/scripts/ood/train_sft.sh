#!/bin/bash
# train_sft.sh

export CUDA_VISIBLE_DEVICES="0,1"

DATA_PATH=/data/ljx/visualRft/share_data/Ship_OpenSet_5shot
MODEL_PATH=/data/ljx/Qwen2-VL-2B-Instruct
SAVE_PATH=/data/ljx/visualRft/share_models/ood/Ship_OpenSet_SFT_5shot_nocot

torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    /data/ljx/visualRft/src/virft/src/sft_openset.py \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_path ${DATA_PATH} \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 40 \
    --max_steps 120 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --max_pixels 401408 \
    --logging_steps 5 \
    --save_steps 50
