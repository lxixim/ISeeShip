#!/bin/bash

export DEBUG_MODE="true"
export LOG_PATH="./debug_grpo_openset_5shot_feature_reward.txt"
export CUDA_VISIBLE_DEVICES="0,1"

DATA_PATH=/data/ljx/visualRft/share_data/Ship_OpenSet_5shot_cot
CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct
SAVE_PATH=/data/ljx/visualRft/share_models/ood/Qwen2-VL-Ship_OpenSet_GRPO_5shotV44
torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    /data/ljx/visualRft/src/virft/src/open_r1/grpo_classification_ood_openset.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --use_peft true \
    --lora_r 512 \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj visual.*.q_proj visual.*.v_proj visual.*.k_proj visual.*.o_proj \
    --deepspeed /data/ljx/visualRft/src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --bf16 true \
    --num_train_epochs 8 \
    --logging_steps 1 \
    --report_to none \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4 \
    --max_steps 200 \
    --max_pixels 401408
