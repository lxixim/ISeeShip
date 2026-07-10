export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_cls_ship30_10_shot.txt"
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES="0,1"
export DATA_PATH=/data/ljx/visualRft/share_data/ViRFT_CLS_ship30_10shot_cot_new
export CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct
export SAVE_PATH=/data/ljx/visualRft/share_models/classification/Qwen2-VL-2B-Instruct_GRPO_ship_cls_10shot

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /data/ljx/visualRft/src/virft/src/open_r1/grpo_classification_reward.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --use_peft true \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_r 512 \
    --lora_target_modules q_proj v_proj k_proj o_proj visual.*.q_proj visual.*.v_proj visual.*.k_proj visual.*.o_proj \
    --dataset_name ${DATA_PATH} \
    --deepspeed /data/ljx/visualRft/src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --learning_rate 2.0e-5 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B_GRPO_ship30_4shot_from_sft \
    --save_steps 50 \
    --max_steps 200 \
    --save_only_model true \
    --num_generations 4