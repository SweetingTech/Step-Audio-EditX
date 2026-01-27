import os
import io
import base64
import numpy as np
import soundfile as sf
import torch
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 引入你的模型库 (基于第一段代码)
from urgent2026_sqa.infer import load_model, infer_single

app = FastAPI()

# ============================
# 1. 环境与设备配置 (参考第二段代码)
# ============================
# 获取 GPU ID，方便多卡部署
worker_id = int(os.environ.get('LOCAL_RANK', 0))
num_gpus = torch.cuda.device_count()
# 如果有 GPU 则使用对应的 ID，否则使用 CPU
device = f"cuda:{worker_id % num_gpus}" if num_gpus > 0 else "cpu"
gpu_lock = Lock()

print(f"Worker ID: {worker_id}, Using Device: {device}")

# ============================
# 2. 加载 Deepfake 模型
# ============================
MODEL_PATH = ""

print(f"Loading Deepfake model from {MODEL_PATH}...")
try:
    # 加载模型 (假设 load_model 内部处理了 device，或者加载后默认在 cpu/gpu)
    # 如果 load_model 支持 device 参数建议传入，例如: load_model(path, device=device)
    model, config = load_model(MODEL_PATH)
    
    # 如果模型是标准的 torch.nn.Module，尝试将其移动到指定设备
    if hasattr(model, "to") and num_gpus > 0:
        model = model.to(device)
        print(f"Model moved to {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# ============================
# 3. 定义请求数据结构
# ============================
class DeepfakeRequest(BaseModel):
    audio_base64: str
    # 如果需要传递其他参数（如文件名用于日志），可以在此添加
    # filename: str = "unknown" 

# ============================
# 4. 音频预处理函数 (基于第一段代码的核心逻辑)
# ============================
def preprocess_base64_to_numpy(base64_str: str):
    try:
        # 1. 解码 Base64
        decoded_bytes = base64.b64decode(base64_str)
        
        # 2. 包装为内存文件
        memory_file = io.BytesIO(decoded_bytes)
        
        # 3. 使用 soundfile 读取
        data, samplerate = sf.read(memory_file)
        
        # 4. 数据类型转换 (Float32) 和 维度调整 (增加 batch 维度: [1, T])
        data = data.astype(np.float32)
        if data.ndim == 1:
            data = data[np.newaxis, :]
            
        return data, samplerate
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None

# ============================
# 5. API 接口定义
# ============================
@app.post("/reward/mos")
def get_mos_reward(req: DeepfakeRequest):
    result = None
    
    # 1. 数据预处理
    audio_array, sr = preprocess_base64_to_numpy(req.audio_base64)
    
    if audio_array is None:
        raise HTTPException(status_code=400, detail="Invalid audio data or decoding failed")

    try:
        # 2. 模型推理 (加锁防止显存冲突或线程不安全)
        with gpu_lock:
            # 使用第一段代码中验证过的 infer_single(model, config, audio_array, sr) 方式
            # 注意：如果 model 在 GPU 上，infer_single 内部可能需要处理 tensor 的设备
            result = infer_single(model, config, audio_array, sr)
            
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 3. 返回结果
    # 假设 result 是一个字典、浮点数或列表，FastAPI 会自动转换为 JSON
    # 如果 result 包含 numpy 类型，可能需要转换，例如 result.item()
    return {
        "status": "success",
        "mos_reward": result['mos']
    }

if __name__ == "__main__":
    import uvicorn
    # 端口设置为 8003 避免冲突，也可改回 8002
    uvicorn.run(app, host="0.0.0.0", port=8003)