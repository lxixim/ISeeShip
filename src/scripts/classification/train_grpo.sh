#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
prepare_training src/open_r1/grpo_classification_reward.py
require_deepspeed
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-./debug_log_classification_grpo.txt}"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT:-12345}" \
    "$TRAIN_ENTRY" \
    --output_dir "$SAVE_PATH" \
    --model_name_or_path "$CKPT_PATH" \
    --use_peft true \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_r "$LORA_R" \
    --lora_target_modules q_proj v_proj k_proj o_proj 'visual.*.q_proj' 'visual.*.v_proj' 'visual.*.k_proj' 'visual.*.o_proj' \
    --dataset_name "$DATA_PATH" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --learning_rate 2.0e-5 \
    --optim adamw_torch \
    --seed "$TRAINING_SEED" \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name "SVI-R1_classification_${SHOT}shot_seed${TRAINING_SEED}" \
    --save_steps 50 \
    --max_steps "$MAX_STEPS" \
    --save_only_model true \
    --num_generations "$GRPO_NUM_GENERATIONS"
