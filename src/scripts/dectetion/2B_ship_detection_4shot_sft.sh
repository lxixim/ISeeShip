export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_sft_ship_detection_5shot.txt"
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES="0,1"
export DATA_PATH=/data/ljx/visualRft/share_data/ship30_coco/shot_5_cot_new
export CKPT_PATH=/data/ljx/Qwen2-VL-2B-Instruct
export SAVE_PATH=/data/ljx/visualRft/share_models/dectetion/Qwen2-VL-2B-Instruct_sft_ship_detection_5shot

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /data/ljx/visualRft/src/virft/src/open_r1/sft_vision_detection.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_path ${DATA_PATH} \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_r 512 \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --save_steps 64 \
    --max_steps 200 \
    --max_length 2048 \
    --learning_rate 2.0e-5 \
    --deepspeed /data/ljx/visualRft/src/virft/local_scripts/zero3.json










