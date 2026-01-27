#!/bin/bash

# --- Path Configuration ---
MODEL_PATH="{EDITX_PATH}"
# --- [Supports multiple data files; please use array format] ---
DATA_FILES=(
    "{TRAINING_INDEX_FILE}"
    # "xxxxx"
    # Additional files can be added here...
)
OUTPUT_DIR="{YOUR_PATH_TO_SAVE_CHECKPOINT}"
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"
CONFIG_PATH="./config/train_config/accelerate_configs/deepspeed_zero2.yaml"

# --- Auto-generate Timestamp Suffix ---
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOG_ROOT}/training_log_${TIMESTAMP}.txt"

# --- Directory Creation ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_ROOT"

# --- Hardware & Distributed Parameters ---
GPU_NUM=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# --- Log Basic Information (Using tee -a to output to both screen and log file) ---
{
    echo "------------------------------------------------"
    echo "Starting training at: $(date)"
    echo "Log file: $LOG_FILE"
    echo "Output Dir: $OUTPUT_DIR"
    echo "------------------------------------------------"
} | tee -a "$LOG_FILE"

REWARD_FUNCS="my_genrm"
SERVER_IP="127.0.0.1"

# --- Construct Command Array ---
# Using an array keeps parameters organized and makes printing easier
CMD_ARGS=(
    accelerate launch
    --config_file "${CONFIG_PATH}"
    --num_processes ${GPU_NUM}
    --main_process_port ${MASTER_PORT}
    src/train_edit.py
    --model_name_or_path "$MODEL_PATH"
    --data_files "${DATA_FILES[@]}"
    --output_dir "$OUTPUT_DIR"
    --reward_server_ip "$SERVER_IP"
    --reward_server_num 4
    --reward_funcs "$REWARD_FUNCS"
    --max_text_length 512
    --max_audio_tokens 1024
    --run_name "Edit-GRPO-Gemini"
    --num_train_epochs 1
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 2
    --learning_rate 1e-6
    --lr_scheduler_type "cosine"
    --warmup_ratio 0.03
    --logging_steps 1
    --save_steps 50
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

# --- Write full execution parameters to log ---
echo -e "\n[Executing Command]:" >> "$LOG_FILE"
echo "${CMD_ARGS[*]}" >> "$LOG_FILE"
echo -e "------------------------------------------------\n" >> "$LOG_FILE"

# --- Execute Training ---
"${CMD_ARGS[@]}" >> "$LOG_FILE" 2>&1
# "${CMD_ARGS[@]}"

# Check Exit Status
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training completed successfully. Log: $LOG_FILE" | tee -a "$LOG_FILE"
else
    echo "Training failed with status $EXIT_STATUS. Please check log: $LOG_FILE" | tee -a "$LOG_FILE"
fi