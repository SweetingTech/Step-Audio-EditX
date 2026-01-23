import os
import torch
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from funasr import AutoModel
from threading import Lock

app = FastAPI()

# worker_id = int(os.environ.get('WORKER_ID', 0)) if 'WORKER_ID' in os.environ else (os.getpid() % 8)
worker_id = int(os.environ.get('LOCAL_RANK', 0)) 
num_gpus = torch.cuda.device_count()
local_rank = worker_id % num_gpus
gpu_lock = Lock()
print("emo: ", local_rank)

# 加载模型
print(f"Loading EMO model on cuda:{local_rank}...")
emo_model = AutoModel(
    model="/mnt/wangyuhao/Model_weights/emotion2vec_plus_large", 
    disable_pbar=True, 
    disable_update=True, 
    device=f"cuda:{local_rank}"
)

class EmoRequest(BaseModel):
    audio_base64: str
    emotion: int

@app.post("/reward/emo")
def get_emo_reward(req: EmoRequest):
    emo_reward = 0
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        with gpu_lock:
            res = emo_model.generate(audio_bytes, output_dir="./tmp", granularity="utterance", extract_embedding=False)
            
            if req.emotion != -1:
                emo_reward = res[0]['scores'][req.emotion]
        
    except Exception as e:
        print(f"EMO reward error: {e}")

    return {
        "emo_reward": emo_reward,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)