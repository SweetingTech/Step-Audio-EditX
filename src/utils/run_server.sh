#!/bin/bash

# ==========================================
# 控制开关：修改此处变量来决定启动哪些服务
# 选项包括: "flow" "emo" "cer" "sim"
# 示例 1 (只启动 flow): ENABLED_SERVICES="flow"
# 示例 2 (全部启动):    ENABLED_SERVICES="flow emo cer sim"
# ==========================================
ENABLED_SERVICES="flow"

# --- 参数配置 ---
NUM_SERVERS=4          # 每种服务启动的实例数量
MAX_GPUS=4             # 机器总共拥有的 GPU 数量

# --- 端口配置 (Base Ports) ---
PORT_BASE_EMO=8100     # EMO Reward 起始端口
PORT_BASE_CER=8200     # CER Reward 起始端口
PORT_BASE_SIM=8300     # SIM Reward 起始端口
PORT_BASE_MOS=8400     # MOS Reward 起始端口
PORT_BASE_FLOW=8080    # Flow Server 起始端口

# --- 日志目录 ---
LOG_DIR="./reward_logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Config: Starting $NUM_SERVERS instances per service."
echo "Enabled Services: $ENABLED_SERVICES"
echo "Logs directory: $LOG_DIR"
echo "========================================================"

# 循环启动所有服务
for ((i=0; i<NUM_SERVERS; i++)); do
  
  # 计算 GPU 编号 (Round-Robin 策略)
  GPU_ID=$((i % MAX_GPUS))

  # ==========================================
  # 1. 启动 EMO Server (如果启用)
  # ==========================================
  if [[ "$ENABLED_SERVICES" == *"emo"* ]]; then
      PORT_EMO=$((PORT_BASE_EMO + i))
      echo "[EMO] Starting instance $i on Port $PORT_EMO, GPU $GPU_ID"
      
      LOCAL_RANK=$GPU_ID uvicorn reward_emo:app \
        --host 0.0.0.0 \
        --port $PORT_EMO \
        --log-level warning > "${LOG_DIR}/log_emo_${PORT_EMO}.txt" 2>&1 &
  fi

  # ==========================================
  # 2. 启动 CER Server (如果启用)
  # ==========================================
  if [[ "$ENABLED_SERVICES" == *"cer"* ]]; then
      PORT_CER=$((PORT_BASE_CER + i))
      echo "[CER] Starting instance $i on Port $PORT_CER, GPU $GPU_ID"
      
      LOCAL_RANK=$GPU_ID uvicorn reward_cer:app \
        --host 0.0.0.0 \
        --port $PORT_CER \
        --log-level warning > "${LOG_DIR}/log_cer_${PORT_CER}.txt" 2>&1 &
  fi

  # ==========================================
  # 3. 启动 SIM Server (如果启用)
  # ==========================================
  if [[ "$ENABLED_SERVICES" == *"sim"* ]]; then
      PORT_SIM=$((PORT_BASE_SIM + i))
      echo "[SIM] Starting instance $i on Port $PORT_SIM, GPU $GPU_ID"
      
      LOCAL_RANK=$GPU_ID uvicorn reward_sim:app \
        --host 0.0.0.0 \
        --port $PORT_SIM \
        --log-level warning > "${LOG_DIR}/log_sim_${PORT_SIM}.txt" 2>&1 &
  fi

  # ==========================================
  # 4. 启动 MOS Server (如果启用)
  # ==========================================
  if [[ "$ENABLED_SERVICES" == *"mos"* ]]; then
      PORT_SIM=$((PORT_BASE_MOS + i))
      echo "[SIM] Starting instance $i on Port $PORT_SIM, GPU $GPU_ID"
      
      LOCAL_RANK=$GPU_ID uvicorn reward_mos:app \
        --host 0.0.0.0 \
        --port $PORT_SIM \
        --log-level warning > "${LOG_DIR}/log_sim_${PORT_SIM}.txt" 2>&1 &
  fi

  # ==========================================
  # 5. 启动 Flow Server (如果启用)
  # ==========================================
  if [[ "$ENABLED_SERVICES" == *"flow"* ]]; then
      PORT_FLOW=$((PORT_BASE_FLOW + i))
      echo "[Flow] Starting instance $i on Port $PORT_FLOW, GPU $GPU_ID"
      
      LOCAL_RANK=$GPU_ID uvicorn flow_server:app \
        --host 0.0.0.0 \
        --port $PORT_FLOW \
        --log-level warning > "${LOG_DIR}/log_flow_${PORT_FLOW}.txt" 2>&1 &
  fi

done

echo "------------------------------------------------"
echo "Startup process completed."

# 仅打印已启用服务的端口信息
if [[ "$ENABLED_SERVICES" == *"emo"* ]]; then
    echo "EMO Ports:  $PORT_BASE_EMO - $((PORT_BASE_EMO + NUM_SERVERS - 1))"
fi
if [[ "$ENABLED_SERVICES" == *"cer"* ]]; then
    echo "CER Ports:  $PORT_BASE_CER - $((PORT_BASE_CER + NUM_SERVERS - 1))"
fi
if [[ "$ENABLED_SERVICES" == *"sim"* ]]; then
    echo "SIM Ports:  $PORT_BASE_SIM - $((PORT_BASE_SIM + NUM_SERVERS - 1))"
fi
if [[ "$ENABLED_SERVICES" == *"mos"* ]]; then
    echo "MOS Ports:  $PORT_BASE_MOS - $((PORT_BASE_MOS + NUM_SERVERS - 1))"
fi
if [[ "$ENABLED_SERVICES" == *"flow"* ]]; then
    echo "Flow Ports: $PORT_BASE_FLOW - $((PORT_BASE_FLOW + NUM_SERVERS - 1))"
fi

echo "Check status with: ps -ef | grep uvicorn"