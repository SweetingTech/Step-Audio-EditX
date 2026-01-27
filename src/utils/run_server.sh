#!/bin/bash

# ============================================================
# Control Switch: Modify this variable to decide which services to start.
# Options include: "flow" "emo" "cer" "sim" "mos"
# Example 1 (Start only flow): ENABLED_SERVICES="flow"
# Example 2 (Start all):       ENABLED_SERVICES="flow emo cer sim mos"
# ============================================================
ENABLED_SERVICES="flow"

# --- Parameter Configuration ---
NUM_SERVERS=4          # Number of instances to start for each enabled service
MAX_GPUS=4             # Total number of GPUs available on the machine

# --- Port Configuration (Base Ports) ---
PORT_BASE_EMO=8100     # Starting port for EMO Reward
PORT_BASE_CER=8200     # Starting port for CER Reward
PORT_BASE_SIM=8300     # Starting port for SIM Reward
PORT_BASE_MOS=8400     # Starting port for MOS Reward
PORT_BASE_FLOW=8080    # Starting port for Flow Server

# --- Log Directory ---
LOG_DIR="./reward_logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Config: Starting $NUM_SERVERS instances per service."
echo "Enabled Services: $ENABLED_SERVICES"
echo "Logs directory: $LOG_DIR"
echo "========================================================"

# Loop to start all services
for ((i=0; i<NUM_SERVERS; i++)); do
  
  # Calculate GPU ID using a Round-Robin strategy
  GPU_ID=$((i % MAX_GPUS))

  # ==========================================
  # 1. Start EMO Server (if enabled)
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
  # 2. Start CER Server (if enabled)
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
  # 3. Start SIM Server (if enabled)
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
  # 4. Start MOS Server (if enabled)
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
  # 5. Start Flow Server (if enabled)
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

# Print port range information for enabled services only
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