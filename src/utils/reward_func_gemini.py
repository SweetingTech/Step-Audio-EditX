import base64
import json
import time
import requests
import logging
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import os
api_key = os.getenv("API_KEY")

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 端口配置 ---
# 仅保留 Flow 用于生成音频，移除其他 Reward Server 端口
PORT_BASE_FLOW = 8080

# --- 常量与 Prompt ---
EMOTIONS = ["admiration", "angry", "confusion", "embarrass", "excited", "fear", "happy", "humour", "sad", "surprised"]

EVAL_PROMPT = """你是一个专业的音频情感验证专家。

### 任务
请听取音频内容，判断说话人的声音情感特征是否符合给定的【目标情感】。

### 输入信息
**目标情感**：{target_emotion}

### 判别标准
1. **忽略语义**：请忽略说话的具体内容，仅关注说话人的语气、音调、语速、音量和能量变化。
2. **特征匹配**：分析音频声学特征是否与【目标情感】的典型表现一致。

### 输出要求
1. 请保持思考过程**简短精炼**，快速判断声音特征与目标情感的匹配度。
2. 必须直接以标准 JSON 格式输出，不要包含 Markdown 标记或其他解释。
3. 如果音频符合目标情感，输出 `true`；如果不符合，输出 `false`。

### 输出示例
{{"is_match": true}}
"""

class GeminiAudioService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 构建标准 Gemini API URL
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.api_key}"

    def call_gemini_api(self, prompt: str, audio_base64: str, max_retries: int = 5, retry_delay: int = 3) -> Optional[bool]:
        """
        使用第一份代码中的通用 REST 结构调用 Gemini API
        """
        if not audio_base64:
            return None

        headers = {"Content-Type": "application/json"}
        
        # 严格按照 Gemini 标准 Payload 结构
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav", # 或者是 audio/mpeg
                            "data": audio_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        for attempt in range(max_retries):
            try:
                # 注意：URL 已经带了 API Key，Header 不需要 Authorization
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    res_json = json.loads(response.text)
                    # 按照第一份代码的路径提取 text
                    try:
                        content = res_json['candidates'][0]['content']['parts'][0]['text']
                        return self._extract_match_result(content)
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Unexpected JSON structure: {e}")
                
                print(f"Attempt {attempt + 1} failed: {response.status_code} - {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Request Exception: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        return None

    def _extract_match_result(self, text_response: str) -> Optional[bool]:
        """解析 Gemini 输出的 JSON 内容"""
        # 清洗常见的 Markdown 标签
        clean_text = text_response.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_text)
            res = data.get("is_match")
            if isinstance(res, bool): return res
            if str(res).lower() == "true": return True
            if str(res).lower() == "false": return False
        except:
            # 正则兜底
            if re.search(r'"is_match"\s*:\s*true', clean_text.lower()): return True
            if re.search(r'"is_match"\s*:\s*false', clean_text.lower()): return False
        return None


# ==========================================
# 工具函数
# ==========================================

def get_balanced_url(base_ip: str, base_port: int, num_servers: int, endpoint: str) -> str:
    """随机负载均衡获取服务器地址"""
    port_offset = random.randint(0, num_servers - 1)
    port = base_port + port_offset
    return f"http://{base_ip}:{port}{endpoint}"

def _get_audio_from_flow(
    uttid: str,
    output_ids: List[int],
    vq0206_codes_vocoder: List[int],
    prompt_wav_path: str,
    server_ip: str,
    num_servers: int,
    proxies: Dict
) -> Optional[str]:
    """请求 Flow Server 生成音频"""
    if output_ids and output_ids[-1] == 3:
        output_ids = output_ids[:-1]
    
    vq_codes = (torch.tensor(vq0206_codes_vocoder) - 65536).tolist()
    
    url = get_balanced_url(server_ip, PORT_BASE_FLOW, num_servers, "/synthesize")
    payload = {
        "uttid": f"{uttid}_{int(time.time()*1000)}",
        "output_ids": output_ids,
        "vq0206_codes_vocoder": vq_codes,
        "prompt_wav_path": prompt_wav_path,
    }

    try:
        resp = requests.post(url, json=payload, proxies=proxies, timeout=60)
        resp.raise_for_status()
        return resp.json().get('audio_base64')
    except Exception as e:
        print(f"[Flow Error] {uttid}: {e}")
        return None

# ==========================================
# 核心处理逻辑 (单样本)
# ==========================================

def _process_sample_for_gemini_reward(
    index: int,
    output_ids: List[int],
    kwargs: Dict,
    server_ip: str,
    num_servers: int,
    gemini_service,
) -> float:
    """
    1. 调用 Flow 生成音频
    2. 调用 Gemini 判断情感
    3. 返回 1.0 或 0.0
    """
    proxies = {"http": None, "https": None}
    
    # --- 1. 生成音频 ---
    audio_b64 = _get_audio_from_flow(
        uttid=f"sample_{index}",
        output_ids=output_ids,
        vq0206_codes_vocoder=kwargs['source_vq02vq06'],
        prompt_wav_path=kwargs['source_audio'],
        server_ip=server_ip,
        num_servers=num_servers,
        proxies=proxies
    )
    
    if not audio_b64:
        return 0.0 # 音频生成失败

    # --- 2. 准备 Gemini Prompt ---
    target_emotion = kwargs.get('target_emotion_text', 'unknown')
    if not target_emotion:
        return 0.0
        
    prompt = EVAL_PROMPT.format(target_emotion=target_emotion)

    # --- 3. 调用 Gemini ---
    is_match = gemini_service.call_gemini_api(prompt, audio_b64)

    # --- 4. 计算分数 ---
    if is_match is True:
        return 1.0
    elif is_match is False:
        return 0.0
    else:
        # 如果 API 调用失败或解析失败，通常给 0 或者一个极小的惩罚值，这里给 0
        return 0.0

# ==========================================
# 主奖励函数入口
# ==========================================

def _parse_kwargs_for_gemini(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """解析参数，提取每条数据的目标情感文本"""
    
    source_audios = reward_kwargs.get('source_audio', [""] * batch_size)
    source_vqs = reward_kwargs.get('source_vq02vq06', [[]] * batch_size)
    
    # 这里的 edit_info 通常是 "Make the voice sound happy" 或直接是 "happy"
    # 我们尽量提取出核心情感词，或者直接将 edit_info 传给 Gemini 让它理解
    edit_infos = reward_kwargs.get('edit_info', [""] * batch_size)
    labels = reward_kwargs.get('label', [""] * batch_size) # 有些数据集字段叫 label
    
    parsed_list = []
    for i in range(batch_size):
        # 优先使用 label，如果没有则尝试从 edit_info 提取
        target_text = labels[i] if labels[i] else edit_infos[i]
        
        # 简单清洗：如果是 "Make the voice sound happy"，Gemini 也能懂
        # 但为了 prompt 准确，最好是 "happy"
        # 这里做一个简单的匹配优化，如果 target_text 包含标准情感词，提取出来
        # 如果不包含，就直接把整个句子传给 Gemini，Gemini 理解能力很强
        final_emotion_target = str(target_text)
        
        info_lower = str(target_text).lower()
        for emo in EMOTIONS:
            if emo in info_lower:
                final_emotion_target = emo
                break
        
        parsed_list.append({
            "source_audio": source_audios[i],
            "source_vq02vq06": source_vqs[i],
            "target_emotion_text": final_emotion_target
        })
    return parsed_list

def gemini_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    server_ip: str = "127.0.0.1",
    num_servers: int = 2,
    gemini_api_key: str = api_key, # 必须传入
    **reward_kwargs
) -> List[float]:
    """
    使用 Gemini 作为 Judge 的奖励函数
    """
    if not gemini_api_key:
        logger.error("Gemini API Key is missing!")
        return [0.0] * len(completion_ids)

    batch_size = len(completion_ids)
    parsed_kwargs = _parse_kwargs_for_gemini(reward_kwargs, batch_size)
    
    # 初始化 Gemini Service (无状态，可复用)
    gemini_service = GeminiAudioService(api_key=gemini_api_key)

    results = [0.0] * batch_size
    # 限制并发数，防止触发 API Rate Limit
    # Gemini Pro Vision 并发通常有限制，建议设为 4-8，具体视 API 额度而定
    max_workers = min(batch_size, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batch_size):
            f = executor.submit(
                _process_sample_for_gemini_reward,
                index=i,
                output_ids=completion_ids[i],
                kwargs=parsed_kwargs[i],
                server_ip=server_ip,
                num_servers=num_servers,
                gemini_service=gemini_service
            )
            futures.append(f)
        
        for i, f in enumerate(futures):
            try:
                results[i] = f.result()
            except Exception as e:
                logger.error(f"Sample {i} failed: {e}")
                results[i] = 0.0
            
    return results

