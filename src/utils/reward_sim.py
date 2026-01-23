import os
import torch
import torchaudio
import base64
import io
from fastapi import FastAPI
from pydantic import BaseModel
from run_sim import get_sim_model, verification2
from threading import Lock

app = FastAPI()
sample_rate = 24000

# GPU 分配逻辑
# worker_id = int(os.environ.get('WORKER_ID', 0)) if 'WORKER_ID' in os.environ else (os.getpid() % 8)
worker_id = int(os.environ.get('LOCAL_RANK', 0)) 
num_gpus = torch.cuda.device_count()
local_rank = worker_id % num_gpus
device = torch.device("cuda", local_rank)
print("sim: ", local_rank)
gpu_lock = Lock()

# 加载模型
print(f"Loading SIM model on cuda:{local_rank}...")
sim_model = get_sim_model(device)

class SimRequest(BaseModel):
    audio_base64: str
    target_audio: str

def decode_base64_audio(b64_string: str):
    audio_bytes = base64.b64decode(b64_string)
    buffer = io.BytesIO(audio_bytes)
    audio, sr = torchaudio.load(buffer)
    return audio, sr

@app.post("/reward/sim")
def get_sim_reward(req: SimRequest):
    sim_reward = 0
    try:
        response_audio, _ = decode_base64_audio(req.audio_base64)
        
        if req.target_audio and req.target_audio != 'None':
            target_audio, file_sr = torchaudio.load(req.target_audio)
            # 重采样
            if file_sr != sample_rate:
                target_audio = torchaudio.functional.resample(target_audio, file_sr, sample_rate)
            if target_audio.shape[0] > 1:
                target_audio = target_audio.mean(dim=0, keepdim=True)
            
            # 推理
            with gpu_lock:
                sim = verification2(response_audio, target_audio, sim_model).item()
                sim_reward = (1 + sim) / 2
            
    except Exception as e:
        print(f"SIM reward error: {e}")

    return {"sim_reward": sim_reward}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)