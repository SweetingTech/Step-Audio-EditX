import os
import time
import json
import random
import requests
import torch
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial, update_wrapper

# --- 端口配置 (需与 run_server_split.sh 一致) ---
PORT_BASE_EMO = 8100
PORT_BASE_CER = 8200
PORT_BASE_SIM = 8300
PORT_BASE_MOS = 8400
PORT_BASE_FLOW = 8080

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
# 新增: 独立获取各类 Reward 的函数
# ==========================================

def _get_cer_reward_from_server(audio_b64: str, ref_text: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """从服务器获取 CER 奖励"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_CER, num_servers, "/reward/cer")
        resp = requests.post(url, json={"audio_base64": audio_b64, "ref_text": ref_text}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("cer_reward", 0.0)
    except Exception as e:
        print(f"[CER Reward Error]: {e}")
        return 0.0

def _get_sim_reward_from_server(audio_b64: str, target_audio: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """从服务器获取 SIM 奖励"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_SIM, num_servers, "/reward/sim")
        resp = requests.post(url, json={"audio_base64": audio_b64, "target_audio": target_audio}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("sim_reward", 0.0)
    except Exception as e:
        print(f"[SIM Reward Error]: {e}")
        return 0.0

def _get_mos_reward_from_server(audio_b64: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """从服务器获取 SIM 奖励"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_MOS, num_servers, "/reward/mos")
        resp = requests.post(url, json={"audio_base64": audio_b64}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("mos_reward", 0.0)
    except Exception as e:
        print(f"[MOS Reward Error]: {e}")
        return 0.0

def _get_emo_reward_from_server(audio_b64: str, emotion_id: int, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """从服务器获取 EMO 奖励"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_EMO, num_servers, "/reward/emo")
        resp = requests.post(url, json={"audio_base64": audio_b64, "emotion": emotion_id}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("emo_reward", 0.0)
    except Exception as e:
        print(f"[EMO Reward Error]: {e}")
        return 0.0

# ==========================================
# 重构后的核心奖励计算逻辑 (单样本)
# ==========================================
def _process_sample_for_reward(
    index: int,
    output_ids: List[int],
    kwargs: Dict,
    reward_type: str, # <--- 关键的新增参数
    server_ip: str,
    num_servers: int,
) -> float:
    """
    处理单条数据，先生成音频，然后根据 reward_type 请求特定的 Reward Server。
    """
    proxies = {"http": None, "https": None}
    
    # 1. 生成音频 (Flow) - 这是公共步骤
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
        return 0.0 # 音频生成失败，奖励为0

    # 2. 根据 reward_type 请求特定的奖励服务器
    reward = 0.0
    if reward_type == 'cer':
        reward = _get_cer_reward_from_server(audio_b64, kwargs['audio_text'], server_ip, num_servers, proxies)
    elif reward_type == 'sim':
        reward = _get_sim_reward_from_server(audio_b64, kwargs['target_audio'], server_ip, num_servers, proxies)
    elif reward_type == 'emo':
        reward = _get_emo_reward_from_server(audio_b64, kwargs['emotion_id'], server_ip, num_servers, proxies)
    elif reward_type == 'mos':
        reward = _get_mos_reward_from_server(audio_b64, server_ip, num_servers, proxies)
    else:
        print(f"[Warning] Unknown reward_type: {reward_type}")

    return reward

# ==========================================
# GRPO 调用的三个主奖励函数
# ==========================================

def _parse_common_kwargs(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """解析参数并转换 Emotion (此函数无需修改)"""
    EMO2ID = {"angry": 0, "fear": 1, "excited": 2, "sad": 3, "surprised": 4}
    
    audio_texts = reward_kwargs.get('audio_text', [""] * batch_size)
    source_audios = reward_kwargs.get('source_audio', [""] * batch_size)
    source_vqs = reward_kwargs.get('source_vq02vq06', [[]] * batch_size)
    target_audios = reward_kwargs.get('target_audio', [""] * batch_size)
    edit_infos = reward_kwargs.get('edit_info', [""] * batch_size)
    
    parsed_list = []
    for i in range(batch_size):
        info = edit_infos[i]
        matched_id = -1
        if info and isinstance(info, str):
            info_lower = info.lower()
            for emo_name, emo_id in EMO2ID.items():
                if emo_name in info_lower:
                    matched_id = emo_id
                    break
        
        parsed_list.append({
            "audio_text": audio_texts[i],
            "source_audio": source_audios[i],
            "source_vq02vq06": source_vqs[i],
            "target_audio": target_audios[i],
            "emotion_id": matched_id
        })
    return parsed_list

def generic_reward_function(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    reward_type: str,  # 'cer', 'sim', 'emo'
    server_ip: str = "127.0.0.1",
    num_servers: int = 2,
    **reward_kwargs
) -> List[float]:
    """通用的奖励函数入口 (修改了内部逻辑)"""
    batch_size = len(completion_ids)
    parsed_kwargs = _parse_common_kwargs(reward_kwargs, batch_size)
    
    results = [0.0] * batch_size
    max_workers = min(batch_size, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batch_size):
            f = executor.submit(
                _process_sample_for_reward, # <-- 调用新的、更高效的函数
                index=i,
                output_ids=completion_ids[i],
                kwargs=parsed_kwargs[i],
                reward_type=reward_type, # <-- 传递 reward_type
                server_ip=server_ip,
                num_servers=num_servers
            )
            futures.append(f)
        
        # 现在 f.result()直接返回一个 float，不再是字典
        for i, f in enumerate(futures):
            results[i] = f.result()
            
    return results

# ==========================================
# 导出给 Trainer 的具体函数 (这些函数无需修改)
# ==========================================

def cer_reward_func(prompts, completions, completion_ids, **kwargs):
    server_ip = kwargs.pop('server_ip', "127.0.0.1")
    num_servers = kwargs.pop('num_servers', 2)
    return generic_reward_function(prompts, completions, completion_ids, 'cer', server_ip, num_servers, **kwargs)

def sim_reward_func(prompts, completions, completion_ids, **kwargs):
    server_ip = kwargs.pop('server_ip', "127.0.0.1")
    num_servers = kwargs.pop('num_servers', 2)
    return generic_reward_function(prompts, completions, completion_ids, 'sim', server_ip, num_servers, **kwargs)

def emo_reward_func(prompts, completions, completion_ids, **kwargs):
    server_ip = kwargs.pop('server_ip', "127.0.0.1")
    num_servers = kwargs.pop('num_servers', 2)
    return generic_reward_function(prompts, completions, completion_ids, 'emo', server_ip, num_servers, **kwargs)

def mos_reward_func(prompts, completions, completion_ids, **kwargs):
    server_ip = kwargs.pop('server_ip', "127.0.0.1")
    num_servers = kwargs.pop('num_servers', 2)
    return generic_reward_function(prompts, completions, completion_ids, 'mos', server_ip, num_servers, **kwargs)