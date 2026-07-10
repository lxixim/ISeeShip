#!/bin/bash
# SFT training script for ship classification - 2 shot (IAD-R1 Expert Flow)

export CUDA_VISIBLE_DEVICES="0,1"
export DATA_PATH=/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_10shot_cot_new
export CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct
export SAVE_PATH=/data/ljx/visualRft/share_models/classification/Qwen2-VL-2B-Instruct_SFT_ship_cls_10shot

mkdir -p ${SAVE_PATH}

# 注意：master_port 确保不与之前的任务冲突
torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12316" \
    /data/ljx/visualRft/src/virft/src/open_r1/sft_vision.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --use_peft true \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_r 128 \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --dataset_name ${DATA_PATH} \
    --dataset_train_split train \
    --deepspeed /data/ljx/visualRft/src/virft/local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 4 \
    --learning_rate 2.0e-5 \
    --run_name Qwen2-VL-2B_SFT_ship30_4shot \
    --max_steps 200 \
    --save_steps 50 \
    --save_only_model true \
    --eval_strategy "no" \
    --ignore_data_skip true \
    --overwrite_output_dir true