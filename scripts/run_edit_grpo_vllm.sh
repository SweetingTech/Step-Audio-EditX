#!/bin/bash

# --- 路径配置 ---
MODEL_PATH="{EDITX_PATH}"
# --- [支持多个数据文件，请使用数组格式] ---
DATA_FILES=(
    "{TRAINING_INDEX_FILE}"
    # "xxxxx"
    # 可以继续添加更多文件...
)
OUTPUT_DIR="{YOUR_PATH_TO_SAVE_CHECKPOINT}"
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"
CONFIG_PATH="./config/train_config/accelerate_configs/deepspeed_zero2.yaml"

# --- 自动生成时间后缀 ---
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOG_ROOT}/training_log_${TIMESTAMP}.txt"

# --- 目录创建 ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_ROOT"

# --- 硬件与分布式参数 ---
GPU_NUM=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# --- 记录基础信息 (使用 tee -a 同时输出到屏幕和日志文件) ---
{
    echo "------------------------------------------------"
    echo "Starting training at: $(date)"
    echo "Log file: $LOG_FILE"
    echo "Output Dir: $OUTPUT_DIR"
    echo "------------------------------------------------"
} | tee -a "$LOG_FILE"

REWARD_FUNCS="my_genrm"
SERVER_IP="127.0.0.1"

# --- 构建命令数组 ---
# 使用数组可以清晰地管理参数，并且方便打印
CMD_ARGS=(
    accelerate launch
    --config_file "${CONFIG_PATH}"
    --num_processes ${GPU_NUM}
    --main_process_port ${MASTER_PORT}
    src/train_edit.py
    --model_name_or_path "$MODEL_PATH"
    --data_files "${DATA_FILES[@]}"
    --output_dir "$OUTPUT_DIR"
    --use_vllm
    --vllm_mode "colocate"
    --vllm_gpu_memory_utilization 0.3
    --vllm_max_model_length 1536
    --reward_server_ip "$SERVER_IP"
    --reward_server_num 4
    --reward_funcs "$REWARD_FUNCS"
    --max_text_length 512
    --max_audio_tokens 1024
    --run_name "Edit-GRPO-Gemini"
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 1e-6
    --lr_scheduler_type "cosine"
    --warmup_ratio 0.03
    --logging_steps 1
    --save_steps 25
    --save_total_limit 2
    --bf16 true
    --gradient_checkpointing true 
    --num_generations 16
    --temperature 1.0
    --beta 0.1
    --resume_from_checkpoint True
    --report_to "wandb"
    --ddp_timeout 5400
)

# --- 将完整运行参数写入日志 ---
echo -e "\n[Executing Command]:" >> "$LOG_FILE"
echo "${CMD_ARGS[*]}" >> "$LOG_FILE"
echo -e "------------------------------------------------\n" >> "$LOG_FILE"

# --- 执行训练 ---
# "${CMD_ARGS[@]}" 展开数组作为命令执行
# >> "$LOG_FILE" 使用追加模式，保留上面的 Header 信息
# "${CMD_ARGS[@]}" >> "$LOG_FILE" 2>&1
"${CMD_ARGS[@]}"

# 检查退出状态
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training completed successfully. Log: $LOG_FILE" | tee -a "$LOG_FILE"
else
    echo "Training failed with status $EXIT_STATUS. Please check log: $LOG_FILE" | tee -a "$LOG_FILE"
fi