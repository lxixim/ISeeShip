#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
prepare_training src/open_r1/sft_vision.py
require_deepspeed

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT:-12316}" \
    "$TRAIN_ENTRY" \
    --output_dir "$SAVE_PATH" \
    --model_name_or_path "$CKPT_PATH" \
    --use_peft true \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_r "$LORA_R" \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --dataset_name "$DATA_PATH" \
    --dataset_train_split train \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 4 \
    --learning_rate 2.0e-5 \
    --optim adamw_torch \
    --seed "$TRAINING_SEED" \
    --run_name "SVI-SFT_classification_${SHOT}shot_seed${TRAINING_SEED}" \
    --max_steps "$MAX_STEPS" \
    --save_steps 50 \
    --save_only_model true \
    --eval_strategy "no" \
    --ignore_data_skip true \
    --overwrite_output_dir true
