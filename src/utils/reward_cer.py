import os
import torch
import base64
import zhconv
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from funasr import AutoModel
from run_wer import process_one
from threading import Lock
import re

app = FastAPI()

# worker_id = int(os.environ.get('WORKER_ID', 0)) if 'WORKER_ID' in os.environ else (os.getpid() % 8)
worker_id = int(os.environ.get('LOCAL_RANK', 0)) 
num_gpus = torch.cuda.device_count()
local_rank = worker_id % num_gpus
gpu_lock = Lock()
print("cer: ", local_rank)

# 加载模型
print(f"Loading ASR model on cuda:{local_rank}...")
asr_model = AutoModel(
    model="/mnt/wangyuhao/Model_weights/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
    disable_pbar=True, 
    disable_update=True,
    device=f"cuda:{local_rank}"
)

class CerRequest(BaseModel):
    audio_base64: str
    ref_text: str

@app.post("/reward/cer")
def get_cer_reward(req: CerRequest):
    cer_reward = 0
    transcription = ""
    # 匹配中文字符的正则范围
    zh_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 匹配英文字母的正则
    en_pattern = re.compile(r'[a-zA-Z]')
    try:
        with gpu_lock:
            audio_bytes = base64.b64decode(req.audio_base64)
            asr_res = asr_model.generate(input=audio_bytes, batch_size_s=300)
            
            transcription = asr_res[0]["text"]
            has_zh = bool(zh_pattern.search(transcription))
            has_en = bool(en_pattern.search(transcription))

            if has_en and not has_zh:
                _, _, wer, _, _, _ = process_one(transcription, req.ref_text, 'en')
            else:
                transcription = zhconv.convert(transcription, 'zh-cn')
                _, _, wer, _, _, _ = process_one(transcription, req.ref_text, 'zh')
            
            cer_reward = float(np.exp(-2.5 * wer))
        
    except Exception as e:
        print(f"CER reward error: {e}")

    return {
        "cer_reward": cer_reward,
        "transcription": transcription
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)