#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
prepare_training src/open_r1/sft_vision_detection.py
require_deepspeed
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-./debug_log_detection_sft.txt}"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT:-12345}" \
    "$TRAIN_ENTRY" \
    --output_dir "$SAVE_PATH" \
    --model_name_or_path "$CKPT_PATH" \
    --dataset_path "$DATA_PATH" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_r "$LORA_R" \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --save_steps 64 \
    --max_steps "$MAX_STEPS" \
    --max_length 2048 \
    --learning_rate 1.0e-6 \
    --optim adamw_torch \
    --seed "$TRAINING_SEED" \
    --run_name "SVI-SFT_detection_${SHOT}shot_seed${TRAINING_SEED}" \
    --deepspeed "$DEEPSPEED_CONFIG"
