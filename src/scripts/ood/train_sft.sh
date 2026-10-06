#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
prepare_training src/open_r1/sft_vision_ood.py
require_deepspeed

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT:-12347}" \
    "$TRAIN_ENTRY" \
    --model_name_or_path "$CKPT_PATH" \
    --dataset_name "$DATA_PATH" \
    --dataset_train_split train \
    --output_dir "$SAVE_PATH" \
    --use_peft true \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --num_train_epochs 40 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --optim adamw_torch \
    --seed "$TRAINING_SEED" \
    --run_name "SVI-SFT_openset_${SHOT}shot_seed${TRAINING_SEED}" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --max_pixels 401408 \
    --logging_steps 5 \
    --save_steps 50
