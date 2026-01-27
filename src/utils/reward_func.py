import os
import time
import json
import random
import requests
import torch
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial, update_wrapper

# --- Port Configuration (Must match run_server_split.sh) ---
PORT_BASE_EMO = 8100
PORT_BASE_CER = 8200
PORT_BASE_SIM = 8300
PORT_BASE_MOS = 8400
PORT_BASE_FLOW = 8080

def get_balanced_url(base_ip: str, base_port: int, num_servers: int, endpoint: str) -> str:
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
    """Request the Flow Server to generate audio"""
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


def _get_cer_reward_from_server(audio_b64: str, ref_text: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """Fetch Character Error Rate (CER) reward from server"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_CER, num_servers, "/reward/cer")
        resp = requests.post(url, json={"audio_base64": audio_b64, "ref_text": ref_text}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("cer_reward", 0.0)
    except Exception as e:
        print(f"[CER Reward Error]: {e}")
        return 0.0

def _get_sim_reward_from_server(audio_b64: str, target_audio: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """Fetch Speaker Similarity (SIM) reward from server"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_SIM, num_servers, "/reward/sim")
        resp = requests.post(url, json={"audio_base64": audio_b64, "target_audio": target_audio}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("sim_reward", 0.0)
    except Exception as e:
        print(f"[SIM Reward Error]: {e}")
        return 0.0

def _get_mos_reward_from_server(audio_b64: str, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """Fetch Mean Opinion Score (MOS) reward from server"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_MOS, num_servers, "/reward/mos")
        resp = requests.post(url, json={"audio_base64": audio_b64}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("mos_reward", 0.0)
    except Exception as e:
        print(f"[MOS Reward Error]: {e}")
        return 0.0

def _get_emo_reward_from_server(audio_b64: str, emotion_id: int, server_ip: str, num_servers: int, proxies: Dict) -> float:
    """Fetch Emotion (EMO) reward from server"""
    try:
        url = get_balanced_url(server_ip, PORT_BASE_EMO, num_servers, "/reward/emo")
        resp = requests.post(url, json={"audio_base64": audio_b64, "emotion": emotion_id}, proxies=proxies, timeout=30)
        resp.raise_for_status()
        return resp.json().get("emo_reward", 0.0)
    except Exception as e:
        print(f"[EMO Reward Error]: {e}")
        return 0.0

# ==========================================
# Refactored Core Reward Calculation Logic (Single Sample)
# ==========================================
def _process_sample_for_reward(
    index: int,
    output_ids: List[int],
    kwargs: Dict,
    reward_type: str, 
    server_ip: str,
    num_servers: int,
) -> float:
    """
    Process a single data point: first generate audio, then request 
    a specific Reward Server based on the reward_type.
    """
    proxies = {"http": None, "https": None}
    
    # 1. Generate Audio (Flow) - This is a shared common step
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
        return 0.0 

    # 2. Request a specific reward server based on reward_type
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


def _parse_common_kwargs(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
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
    batch_size = len(completion_ids)
    parsed_kwargs = _parse_common_kwargs(reward_kwargs, batch_size)
    
    results = [0.0] * batch_size
    max_workers = min(batch_size, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(batch_size):
            f = executor.submit(
                _process_sample_for_reward,
                index=i,
                output_ids=completion_ids[i],
                kwargs=parsed_kwargs[i],
                reward_type=reward_type,
                server_ip=server_ip,
                num_servers=num_servers
            )
            futures.append(f)
        
        for i, f in enumerate(futures):
            results[i] = f.result()
            
    return results

# ==========================================
# Exported Specific Functions for the Trainer
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