#!/bin/bash
# Zero-shot OOD Detection - 纯ID分类训练 (LoRA版本)
# 训练数据：只有14个ID类别 × 4张 = 56张，0个OOD样本
# 推理时：用置信度检测OOD（confidence < 0.5 → OOD）

export DATA_PATH=/data/ljx/visualRft/share_data/Ship30_Zeroshot_OOD_4shot  ### ✅ 纯ID数据集
export CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct  ### ✅ 从基础模型开始
export SAVE_PATH=/data/ljx/visualRft/share_models/Qwen2-VL-2B-Instruct_GRPO_Zeroshot_OOD_4shot
export CUDA_VISIBLE_DEVICES=0,1

# Debug模式（可选）
export DEBUG_MODE="true"
export LOG_PATH="${SAVE_PATH}/debug_log.txt"

echo "========================================================================"
echo "🔥 Zero-shot OOD Detection Training (LoRA)"
echo "========================================================================"
echo "数据集: ${DATA_PATH}"
echo "  - 训练: 14个ID类别 × 4张 = 56张"
echo "  - 测试: ID + OOD（完全没见过的类别）"
echo ""
echo "初始模型: ${CKPT_PATH}"
echo "输出路径: ${SAVE_PATH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "LoRA配置:"
echo "  - r: 1024"
echo "  - alpha: 1024"
echo "  - dropout: 0.05"
echo ""
echo "策略:"
echo "  ✅ 只用ID类别训练"
echo "  ✅ solution不含confidence（让模型自己学）"
echo "  ✅ 奖励函数: 正确+高conf → 高奖励, 错误+低conf → 部分奖励"
echo "  ✅ 推理时: 高conf → ID, 低conf → OOD"
echo "========================================================================"

torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/virft/src/open_r1/grpo_classification_ood.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --use_peft true \
    --lora_alpha 2048 \
    --lora_dropout 0.05 \
    --lora_r 2048 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj visual.*.q_proj visual.*.v_proj visual.*.k_proj visual.*.o_proj \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --eval_strategy no \
    --run_name Qwen2-VL-2B-ZeroShot-OOD-LoRA \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4 \
    --learning_rate 1e-7

echo ""
echo "========================================================================"
echo "✅ 训练完成！"
echo "========================================================================"
echo "模型保存在: ${SAVE_PATH}"
echo ""
echo "下一步:"
echo "  1. 评估: python evaluate_zeroshot_ood.py --model_path ${SAVE_PATH}/checkpoint-XXX"
echo "  2. 测试: python test_single_image.py --model_path ${SAVE_PATH}/checkpoint-XXX"
echo "========================================================================"