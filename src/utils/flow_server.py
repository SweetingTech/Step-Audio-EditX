import os
import io
import base64
import torch
import torchaudio
import numpy as np
from threading import Lock
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
os.environ['NO_PROXY'] = '*'  # 或者写具体的 IP: '100.101.97.230'
os.environ['no_proxy'] = '*'

FLOW_PATH = "where_you_download_dir/Step-Audio-EditX/CosyVoice-300M-25Hz/"  # 模型路径，最终到 CosyVoice-300M-25Hz 文件夹

# 假设 CosyVoice 类位于此路径下
from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice

app = FastAPI()
SAMPLE_RATE = 24000

# --- 环境变量与设备配置 ---
# worker_id = int(os.environ.get('WORKER_ID', os.getpid() % 8))
worker_id = int(os.environ.get('LOCAL_RANK', 0)) 
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:
    local_rank = worker_id % NUM_GPUS
    DEVICE = torch.device("cuda", local_rank)
else:
    DEVICE = torch.device("cpu")

flow_lock = Lock()

# --- 模型加载 ---
print(f"Worker {worker_id} 正在加载模型到: {DEVICE}...")
try:
    # 路径请根据实际情况修改
    cosy_model = CosyVoice(
        FLOW_PATH, 
        device=DEVICE
    )
    print(f"Worker {worker_id} 模型加载成功。")
except Exception as e:
    print(f"Worker {worker_id} 模型加载失败: {e}")
    cosy_model = None

# --- 辅助函数：处理 Base64 音频 ---
def decode_base64_audio(b64_string: str):
    """将 Base64 字符串解码为 Tensor (C, T) 和采样率"""
    audio_bytes = base64.b64decode(b64_string)
    buffer = io.BytesIO(audio_bytes)
    # 假设传过来的是 wav 格式，如果是 raw pcm 需要用 distinct logic
    audio, sr = torchaudio.load(buffer)
    return audio, sr

# def encode_audio_to_base64(audio_tensor: torch.Tensor, sr: int):
#     """将 Audio Tensor 编码为 Base64 WAV 字符串"""
#     # 确保在 CPU 上
#     audio_tensor = audio_tensor.detach().cpu()
#     if audio_tensor.dim() == 1:
#         audio_tensor = audio_tensor.unsqueeze(0) # (1, T)
        
#     buffer = io.BytesIO()
#     # format="wav" encoding="PCM_S" bits_per_sample=16 
#     # 使用 16-bit PCM 可以进一步减少传输体积 (相比 32-bit float 减少一半)，且听感无损
#     torchaudio.save(buffer, audio_tensor, sr, format="wav", encoding="PCM_S", bits_per_sample=16)
#     return base64.b64encode(buffer.getvalue()).decode('utf-8')

from scipy.io import wavfile

def encode_audio_to_base64(audio_tensor: torch.Tensor, sr: int):
    audio_tensor = audio_tensor.detach().cpu()
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.squeeze(0)
    
    audio_np = audio_tensor.numpy()

    # !!! 关键步骤：Scipy 不会自动做 float -> int16 的转换
    # 假设模型输出在 [-1, 1] 之间
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sr, audio_int16)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# --- 请求体定义 ---
class FlowMatchingRequest(BaseModel):
    # Token 列表传 List[int] 是非常标准且高效的做法
    output_ids: List[int] = Field(..., description="目标音频 Token ID")
    vq0206_codes_vocoder: List[int] = Field(..., description="Vocoder Codes")
    uttid: str = Field(..., description="请求ID")
    
    # 改进点：同时支持 路径 或 Base64 内容
    # 优先使用 audio_data，这样服务端就是无状态的，不需要挂载存储
    prompt_wav_path: Optional[str] = None
    prompt_audio_data: Optional[str] = Field(None, description="Prompt音频的Base64字符串")

@app.post("/synthesize")
def synthesize_audio(req: FlowMatchingRequest):
    if cosy_model is None:
        raise HTTPException(status_code=503, detail="模型未就绪")

    try:
        # 1. 获取 Prompt 音频 (优先内存数据，其次文件路径)
        prompt_wav = None
        prompt_wav_sr = 0
        
        if req.prompt_audio_data:
            prompt_wav, prompt_wav_sr = decode_base64_audio(req.prompt_audio_data)
        elif req.prompt_wav_path and os.path.exists(req.prompt_wav_path):
            prompt_wav, prompt_wav_sr = torchaudio.load(req.prompt_wav_path)
        else:
            raise HTTPException(status_code=400, detail="必须提供 prompt_audio_data 或 有效的 prompt_wav_path")

        # 2. 预处理
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            
        # 简单的音量归一化
        norm = torch.max(torch.abs(prompt_wav), dim=1, keepdim=True)[0]
        if norm > 0.6:
            prompt_wav = prompt_wav / norm * 0.6

        # 3. 推理
        with flow_lock:
            speech_feat, _ = cosy_model.frontend.extract_speech_feat(prompt_wav, prompt_wav_sr)
            speech_embedding = cosy_model.frontend.extract_spk_embedding(prompt_wav, prompt_wav_sr)

            output_ids_tensor = torch.tensor([req.output_ids], dtype=torch.long, device=DEVICE)
            vq_codes_tensor = torch.tensor([req.vq0206_codes_vocoder], dtype=torch.long, device=DEVICE)

            # 生成
            output_audio = cosy_model.token2wav_nonstream(
                output_ids_tensor - 65536,
                vq_codes_tensor,
                speech_feat.to(torch.bfloat16).to(DEVICE),
                speech_embedding.to(torch.bfloat16).to(DEVICE),
            )

        # 4. 返回结果 (关键修改：返回 Base64 而不是 List[float])
        # 使用 16bit wav 编码返回，体积小，解析快
        b64_audio = encode_audio_to_base64(output_audio, SAMPLE_RATE)

        return {
            "status": "success",
            "uttid": req.uttid,
            "audio_base64": b64_audio, # 客户端收到后解码保存即可
            # "data_list": output_audio.tolist() # 强烈建议删除此行，性能杀手
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        # 生产环境建议打印完整 traceback
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))