#!/bin/bash
# GRPO训练脚本

export CUDA_VISIBLE_DEVICES="0,1"

# ===== 路径配置 =====
export SFT_MODEL=/data/ljx/visualRft/share_models/ood/Qwen2-VL-2B-OpenSet-16shot-SFT_LORA128_step1-merge
export DATA_PATH=/data/ljx/visualRft/share_data/Ship30_OpenSet_16shot
export SAVE_PATH=/data/ljx/visualRft/share_models/ood/Qwen2-VL-2B-Instruct_GRPO_16shot_fromsft_LORA128_step1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p ${SAVE_PATH}

echo "========================================================================"
echo "🔥 GRPO Training"
echo "========================================================================"
echo "SFT模型: ${SFT_MODEL}"
echo "数据集:   ${DATA_PATH}"
echo "保存:     ${SAVE_PATH}"
echo "========================================================================"
echo ""

torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    src/virft/src/open_r1/grpo_classification_ood.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${SFT_MODEL} \
    --use_peft true \
    --lora_r 1024 \
    --lora_alpha 2048 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj visual.*.q_proj \
    --dataset_name ${DATA_PATH} \
    --dataset_train_split train \
    --dataset_test_split train \
    --deepspeed src/virft/local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --max_prompt_length 1024 \
    --run_name GRPO-simple \
    --save_steps 32 \
    --save_only_model true \
    --eval_strategy no \
    --num_generations 4 \
    --reward_weights 1.0 1.0 \
    --debug_mode false

echo ""
echo "========================================================================"
echo "✅ 训练完成!"
echo "========================================================================"
echo "下一步: 推理时使用规则计算confidence"
echo "========================================================================"