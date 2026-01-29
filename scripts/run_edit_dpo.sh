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

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="${LOG_ROOT}/dpo_log_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_ROOT"

GPU_NUM=8

CMD_ARGS=(
    accelerate launch
    --config_file "${CONFIG_PATH}"
    --num_processes ${GPU_NUM}
    src/train_edit_dpo.py
    --model_name_or_path "$MODEL_PATH"
    --data_files "${DATA_FILES[@]}"
    --output_dir "$OUTPUT_DIR"
    --max_prompt_length 512
    --max_length 1536
    --num_train_epochs 1
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 2
    --learning_rate 1e-6
    --lr_scheduler_type "cosine"
    --warmup_ratio 0.1
    --beta 0.1
    --logging_steps 1
    --save_steps 100
    --save_total_limit 2
    --bf16 true
    --gradient_checkpointing true 
    --dataset_num_proc 16
    --report_to "wandb"
    --ddp_timeout 5400
    --resume_from_checkpoint True
)

echo "Executing DPO Command..." | tee -a "$LOG_FILE"
# "${CMD_ARGS[@]}" >> "$LOG_FILE" 2>&1
"${CMD_ARGS[@]}" 

EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "DPO completed successfully." | tee -a "$LOG_FILE"
else
    echo "DPO failed. Check log: $LOG_FILE" | tee -a "$LOG_FILE"
fi