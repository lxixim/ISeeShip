#!/bin/bash
# SFT训练 - 开集识别优化版
# 核心改进：防止输出截断，确保格式完整

export CUDA_VISIBLE_DEVICES="0,1"
export DATA_PATH=/data/ljx/visualRft/share_data/Ship30_OpenSet_16shot
export CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct
export SAVE_PATH=/data/ljx/visualRft/share_models/ood/Qwen2-VL-2B-OpenSet-16shot-SFT_LORA128_step1

mkdir -p ${SAVE_PATH}

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12348" \
    /data/ljx/visualRft/src/virft/src/open_r1/sft_vision_ood.py \
    --output_dir ${SAVE_PATH} \
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
    --num_train_epochs 2 \
    --learning_rate 2.0e-5 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --run_name SFT_openset_v3 \
    --save_steps 30 \
    --save_only_model true \
    --eval_strategy "no" \
    --ignore_data_skip true \
    --overwrite_output_dir true

echo ""
echo "========================================================================"
echo "✅ SFT训练完成！"
echo "========================================================================"