import os
import time
import json
import random
import requests
import torch
import logging
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 屏蔽 urllib3 和 requests 的繁杂日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ==========================================
# 常量配置
# ==========================================

# Flow Server 端口 (用于将 token 转为音频)
PORT_BASE_FLOW = 8080

# StepAudio R1 默认配置 (可通过 reward_kwargs 覆盖)
DEFAULT_R1_API_URL = ""
DEFAULT_R1_MODEL_NAME = ""

# 情感列表 (用于简单的关键词匹配)
EMOTIONS = ["admiration", "angry", "confusion", "embarrass", "excited", "fear", "happy", "humour", "sad", "surprised"]

# 评测 Prompt (与你提供的脚本保持一致)
EVAL_PROMPT = """你是一个专业的音频情感验证专家。

### 任务
请听取音频内容，判断说话人的声音情感特征是否符合给定的【目标情感】。

### 输入信息
**目标情感**：{target_emotion}

### 判别标准
1. **忽略语义**：请忽略说话的具体内容，仅关注说话人的语气、音调、语速、音量和能量变化。
2. **特征匹配**：分析音频声学特征是否与【目标情感】的典型表现一致。

### 输出要求
1. 请保持思考过程**简短精炼**，快速判断声音特征与目标情感的匹配度，**严禁冗长的重复论证**。
2. 必须直接以标准 JSON 格式输出，不要包含 Markdown 标记或其他解释。
3. 如果音频符合目标情感，输出 `true`；如果不符合，输出 `false`。

### 输出示例
{{"is_match": true}}
"""

# ==========================================
# StepAudio R1 服务类
# ==========================================

class StepAudioR1Service:
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name

    def _extract_match_result(self, text_response: str) -> Optional[bool]:
        """解析 JSON 输出"""
        if not text_response:
            return None
        try:
            # 尝试直接解析
            clean_text = text_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            res = data.get("is_match")
            if isinstance(res, bool): return res
            return str(res).lower() == "true"
        except:
            # 兜底正则解析
            match = re.search(r'"is_match"\s*:\s*(true|false)', text_response.lower())
            if match:
                return match.group(1) == "true"
        return None

    def call_r1_judge(self, prompt: str, audio_base64: str) -> float:
        """
        调用 StepAudio R1 进行判别
        返回: 1.0 (匹配), 0.0 (不匹配或错误)
        """
        if not audio_base64:
            return 0.0

        # 构造符合 StepAudio API 规范的消息体
        messages = [
            {"role": "system", "content": ""},
            {"role": "human", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio", 
                    "input_audio": {
                        "data": audio_base64, 
                        "format": "wav"
                    }
                },
            ]},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True, # 保持 stream=True 以兼容特定 API 行为
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "stop_token_ids": [151665],
            "skip_special_tokens": True
        }

        full_text = ""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            full_text = ""  # 每次重试前必须清空
            try:
                # 发起请求
                with requests.post(self.api_url, json=payload, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line: continue
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '): 
                                line_str = line_str[6:]
                            if line_str == '[DONE]': 
                                break
                            
                            data = json.loads(line_str)
                            delta = data['choices'][0]['delta']
                            # 兼容不同字段名
                            text = delta.get('tts_content', {}).get('tts_text') or delta.get('content', '')
                            if text:
                                full_text += text
                        except:
                            continue
                
                # 解析结果
                # 如果代码执行到这里，说明网络请求成功，直接解析并返回，不再进行后续循环
                is_match = self._extract_match_result(full_text)
                
                if is_match is True:
                    return 1.0
                elif is_match is False:
                    return 0.0
                else:
                    # API 通了但解析不出 True/False，直接返回 0 (或者你可以选择 raise Exception 让它重试)
                    return 0.0

            except Exception as e:
                logger.error(f"StepAudioR1 API Error (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # 超过最大重试次数，返回默认值
                    return 0.0
# ==========================================
# 音频生成工具函数 (Flow Server)
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
) -> Optional[str]:
    """请求 Flow Server 生成音频"""
    proxies = {"http": None, "https": None}
    
    # 移除结束符
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
        resp = requests.post(url, json=payload, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get('audio_base64')
    except Exception as e:
        logger.error(f"[Flow Error] {uttid}: {e}")
        return None

# ==========================================
# 核心处理逻辑 (单样本)
# ==========================================

def _process_sample_for_r1_reward(
    index: int,
    output_ids: List[int],
    kwargs: Dict,
    server_ip: str,
    num_servers: int,
    r1_service: StepAudioR1Service
) -> float:
    """
    1. Flow 生成音频 (Base64)
    2. StepAudioR1 判别情感
    3. 返回分数
    """
    
    # --- 1. 生成音频 ---
    audio_b64 = _get_audio_from_flow(
        uttid=f"sample_{index}",
        output_ids=output_ids,
        vq0206_codes_vocoder=kwargs['source_vq02vq06'],
        prompt_wav_path=kwargs['source_audio'],
        server_ip=server_ip,
        num_servers=num_servers
    )
    
    if not audio_b64:
        return 0.0 

    # --- 2. 准备 Prompt ---
    target_emotion = kwargs.get('target_emotion_text', 'unknown')
    # 如果没拿到情感词，说明数据有问题，给0分
    if not target_emotion or target_emotion == 'unknown':
        return 0.0
        
    prompt = EVAL_PROMPT.format(target_emotion=target_emotion)

    # --- 3. 调用 R1 Judge ---
    score = r1_service.call_r1_judge(prompt, audio_b64)
    return score

# ==========================================
# 主奖励函数入口
# ==========================================

def _parse_kwargs_for_r1(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """解析参数，提取每条数据的目标情感文本"""
    source_audios = reward_kwargs.get('source_audio', [""] * batch_size)
    source_vqs = reward_kwargs.get('source_vq02vq06', [[]] * batch_size)
    
    # 获取情感标签
    edit_infos = reward_kwargs.get('edit_info', [""] * batch_size)
    labels = reward_kwargs.get('label', [""] * batch_size)
    
    parsed_list = []
    for i in range(batch_size):
        # 优先使用 label，否则尝试 edit_info
        raw_text = labels[i] if labels[i] else edit_infos[i]
        
        # 简单清洗与提取
        final_emotion_target = str(raw_text)
        info_lower = str(raw_text).lower()
        
        # 如果句子中包含标准情感词，优先提取标准词（因为 Prompt 需要精准）
        # 如果没匹配到，则传入原始文本，让模型自己理解
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

def step_audio_r1_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    server_ip: str = "127.0.0.1",
    num_servers: int = 2,
    r1_api_url: str = DEFAULT_R1_API_URL,
    r1_model_name: str = DEFAULT_R1_MODEL_NAME,
    **reward_kwargs
) -> List[float]:
    """
    使用 StepAudio R1 模型作为 Judge 的奖励函数
    """
    batch_size = len(completion_ids)
    parsed_kwargs = _parse_kwargs_for_r1(reward_kwargs, batch_size)
    
    # 初始化服务
    r1_service = StepAudioR1Service(api_url=r1_api_url, model_name=r1_model_name)

    results = [0.0] * batch_size
    # 根据显存和并发能力调整 max_workers
    max_workers = min(batch_size, 8) 

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batch_size):
            f = executor.submit(
                _process_sample_for_r1_reward,
                index=i,
                output_ids=completion_ids[i],
                kwargs=parsed_kwargs[i],
                server_ip=server_ip,
                num_servers=num_servers,
                r1_service=r1_service
            )
            futures.append(f)
        
        for i, f in enumerate(futures):
            try:
                results[i] = f.result()
            except Exception as e:
                logger.error(f"Sample {i} failed in main loop: {e}")
                results[i] = 0.0
            
    return results

