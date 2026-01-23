import os
import time
import json
import random
import requests
import torch
import logging
import re
import tempfile
import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础配置与常量
# ==========================================

# Flow Server 端口配置 (通常不需要修改)
PORT_BASE_FLOW = 8080

# 默认评测 Prompt 模板 (用户可在子类中覆盖)
DEFAULT_EVAL_PROMPT = """你是一个专业的音频评估专家。

### 任务
请听取音频内容，判断其是否符合以下【目标要求】。

### 输入信息
**目标要求**：{target_instruction}

### 判别标准
1. **忽略语义**：除非目标要求特定内容，否则请忽略说话的具体内容，仅关注语气、情感、音质等声学特征。
2. **特征匹配**：分析音频声学特征是否与【目标要求】高度一致。

### 输出要求
1. 必须直接以标准 JSON 格式输出，不要包含 Markdown 标记或其他解释。
2. 包含一个字段 "is_match" (boolean) 或者 "score" (0-1 float)。

### 输出示例
{{"is_match": true}}
"""

# ==========================================
# 2. 抽象基类：Generative RM 接口
# ==========================================

class BaseGenerativeRM(ABC):
    """
    所有 Generative Reward Model 的基类。
    用户需要继承此类并实现 `call_model` 方法。
    """
    def __init__(self, prompt_template: str = DEFAULT_EVAL_PROMPT):
        self.prompt_template = prompt_template

    def format_prompt(self, target_instruction: str) -> str:
        """构建 Prompt"""
        return self.prompt_template.format(target_instruction=target_instruction)

    @abstractmethod
    def call_model(self, prompt: str, audio_base64: str) -> Optional[Union[bool, float]]:
        """
        [需要用户实现]
        调用具体的 LLM/VLM API 进行推理。
        
        参数:
            prompt: 构造好的文本提示词
            audio_base64: 音频文件的 base64 字符串 (无 header)
            
        返回:
            float (0.0 - 1.0) 或 bool (True/False)
            如果调用失败，返回 None
        """
        pass

    def parse_response(self, response_text: str) -> float:
        """
        通用解析逻辑，从 JSON 字符串中提取分数。
        支持 {"is_match": true} 或 {"score": 0.8}
        """
        if not response_text:
            return 0.0
            
        # 清洗 markdown 代码块
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(clean_text)
            
            # 优先检查 score 字段
            if "score" in data:
                return float(data["score"])
            
            # 检查 is_match 字段
            res = data.get("is_match")
            if isinstance(res, bool):
                return 1.0 if res else 0.0
            if str(res).lower() == "true": return 1.0
            if str(res).lower() == "false": return 0.0
            
        except json.JSONDecodeError:
            # 兜底正则匹配
            if re.search(r'"is_match"\s*:\s*true', clean_text.lower()): return 1.0
            if re.search(r'"is_match"\s*:\s*false', clean_text.lower()): return 0.0
            
        logger.warning(f"Failed to parse response: {clean_text[:100]}...")
        return 0.0

    def save_temp_audio(self, audio_base64: str, suffix=".wav") -> str:
        """
        [工具函数] 如果模型只接受文件路径，可用此函数将 base64 存为临时文件。
        用法:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(base64.b64decode(audio_base64))
                tmp.flush()
                # call your model with tmp.name
        """
        pass # 仅作为提示，具体在 call_model 中按需实现

# ==========================================
# 3. [用户实现区] 自定义 GenRM
# ==========================================

class CustomGenerativeRM(BaseGenerativeRM):
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = kwargs.get("api_url", "")

    def call_model(self, prompt: str, audio_base64: str) -> Optional[float]:
        """
        TODO: 用户在此处实现具体的 API 调用逻辑 (例如 OpenAI / Gemini / Claude / Local vLLM)
        """
        
        # --- 示例：伪代码结构 ---
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构造 Payload (请根据实际模型 API 格式修改)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        # 假设 API 支持直接传 base64 (如 Gemini / GPT-4o-audio)
                        # 如果需要 URL 或文件上传，请在此处处理
                        {
                            "type": "input_audio", 
                            "input_audio": {
                                "data": audio_base64, 
                                "format": "wav"
                            }
                        },
                    ]
                }
            ]
        }
        
        try:
            # 发起请求 (这里注释掉，防止直接运行报错)
            # response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            # response_json = response.json()
            # content = response_json['choices'][0]['message']['content']
            
            # --- 模拟返回 (测试用) ---
            # 实际使用时请删除下面这行，并返回 self.parse_response(content)
            import random
            return 1.0 if random.random() > 0.5 else 0.0
            
        except Exception as e:
            logger.error(f"Model Inference Error: {e}")
            return None

# ==========================================
# 4. 基础设施 (Flow Server 交互) - 请勿修改
# ==========================================

def get_balanced_url(base_ip: str, base_port: int, num_servers: int, endpoint: str) -> str:
    """随机负载均衡"""
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
    # 移除特殊的 EOS token (如果有)
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
        logger.error(f"[Flow Error] {uttid}: {e}")
        return None

# ==========================================
# 5. 核心处理流程
# ==========================================

def _process_single_sample(
    index: int,
    output_ids: List[int],
    kwargs: Dict,
    server_ip: str,
    num_servers: int,
    genrm_model: BaseGenerativeRM
) -> float:
    """
    单样本处理流水线:
    Flow生成 -> 构建Prompt -> GenRM打分 -> 解析分数
    """
    proxies = {"http": None, "https": None}
    
    # 1. 提取目标指令 (支持 label 或 edit_info 字段)
    target_instruction = kwargs.get('label') or kwargs.get('edit_info') or "unknown"
    
    # 2. 生成音频
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
        return 0.0 # 生成失败惩罚

    # 3. 准备 Prompt
    prompt = genrm_model.format_prompt(target_instruction=str(target_instruction))

    # 4. 模型推理
    result = genrm_model.call_model(prompt, audio_b64)

    # 5. 返回结果
    if result is None:
        return 0.0 # 推理失败惩罚
    
    # 如果 result 已经是 float/bool，直接处理
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    return float(result)

# ==========================================
# 6. 奖励函数入口
# ==========================================

def _parse_batch_kwargs(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """将 batch 形式的 kwargs 转为 list of dict"""
    keys = ['source_audio', 'source_vq02vq06', 'edit_info', 'label']
    parsed_list = []
    
    # 确保所有 key 都有值，防止 key error
    sanitized_kwargs = {k: reward_kwargs.get(k, [None]*batch_size) for k in keys}
    
    for i in range(batch_size):
        item = {k: sanitized_kwargs[k][i] for k in keys}
        parsed_list.append(item)
    return parsed_list

def genrm_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    server_ip: str = "127.0.0.1",
    num_servers: int = 2,
    # 下面的参数通过 command line args 或 yaml 传入
    genrm_api_key: str = os.getenv("GENRM_API_KEY", ""), 
    genrm_api_url: str = "",
    genrm_model_name: str = "gpt-4o",
    **reward_kwargs
) -> List[float]:
    """
    通用 Generative Reward Function 入口
    """
    batch_size = len(completion_ids)
    parsed_kwargs = _parse_batch_kwargs(reward_kwargs, batch_size)
    
    # 初始化用户自定义的 GenRM
    # 这里的参数可以根据 CustomGenerativeRM 的 __init__ 方法灵活调整
    genrm_model = CustomGenerativeRM(
        api_key=genrm_api_key, 
        api_url=genrm_api_url, 
        model_name=genrm_model_name
    )

    results = [0.0] * batch_size
    # 限制并发数
    max_workers = min(batch_size, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batch_size):
            f = executor.submit(
                _process_single_sample,
                index=i,
                output_ids=completion_ids[i],
                kwargs=parsed_kwargs[i],
                server_ip=server_ip,
                num_servers=num_servers,
                genrm_model=genrm_model
            )
            futures.append(f)
        
        for i, f in enumerate(futures):
            try:
                results[i] = f.result()
            except Exception as e:
                logger.error(f"GenRM Sample {i} unexpected error: {e}")
                results[i] = 0.0
            
    return results