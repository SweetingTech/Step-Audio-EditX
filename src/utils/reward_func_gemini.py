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

# --- Logging Configuration ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Port Configuration ---
# Keep only Flow for audio generation; other Reward Server ports have been removed
PORT_BASE_FLOW = 8080

# --- Constants & Prompts ---
EMOTIONS = ["admiration", "angry", "confusion", "embarrass", "excited", "fear", "happy", "humour", "sad", "surprised"]

EVAL_PROMPT = """You are a professional audio emotion verification expert.

### Task
Please listen to the audio content and determine if the speaker's vocal emotional characteristics match the given [Target Emotion].

### Input Information
**Target Emotion**: {target_emotion}

### Criteria
1. **Ignore Semantics**: Please ignore the specific verbal content. Focus only on the speaker's tone, pitch, tempo, volume, and energy dynamics.
2. **Feature Matching**: Analyze whether the acoustic features of the audio are consistent with the typical expression of the [Target Emotion].

### Output Requirements
1. Keep your thought process **concise and refined**, quickly judging the match between vocal features and target emotion.
2. Output must be directly in standard JSON format; do not include Markdown tags or other explanations.
3. If the audio matches the target emotion, output `true`; if not, output `false`.

### Output Example
{{"is_match": true}}
"""

class GeminiAudioService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 构建标准 Gemini API URL
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.api_key}"

    def call_gemini_api(self, prompt: str, audio_base64: str, max_retries: int = 5, retry_delay: int = 3) -> Optional[bool]:
        """
        Calls the Gemini API using a standard REST structure.
        """
        if not audio_base64:
            return None

        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav", # audio/mpeg
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
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    res_json = json.loads(response.text)
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
        """Parses the JSON content output by Gemini"""
        clean_text = text_response.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_text)
            res = data.get("is_match")
            if isinstance(res, bool): return res
            if str(res).lower() == "true": return True
            if str(res).lower() == "false": return False
        except:
            if re.search(r'"is_match"\s*:\s*true', clean_text.lower()): return True
            if re.search(r'"is_match"\s*:\s*false', clean_text.lower()): return False
        return None


# ==========================================
# Utility Functions
# ==========================================

def get_balanced_url(base_ip: str, base_port: int, num_servers: int, endpoint: str) -> str:
    """Obtains server address via random load balancing"""
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
    """Requests audio generation from the Flow Server"""
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
# Core Processing Logic (Single Sample)
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
    1. Call Flow to generate audio.
    2. Call Gemini to judge emotion.
    3. Return 1.0 or 0.0.
    """
    proxies = {"http": None, "https": None}
    
    # --- 1. Generate Audio ---
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

    # --- 2. Prepare Gemini Prompt ---
    target_emotion = kwargs.get('target_emotion_text', 'unknown')
    if not target_emotion:
        return 0.0
        
    prompt = EVAL_PROMPT.format(target_emotion=target_emotion)

    # --- 3. Call Gemini API ---
    is_match = gemini_service.call_gemini_api(prompt, audio_b64)

    # --- 4. Calculate Score ---
    if is_match is True:
        return 1.0
    elif is_match is False:
        return 0.0
    else:
        return 0.0

# ==========================================
# Main Reward Function Entry Point
# ==========================================

def _parse_kwargs_for_gemini(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """Parses arguments and extracts target emotion text for each entry"""
    
    source_audios = reward_kwargs.get('source_audio', [""] * batch_size)
    source_vqs = reward_kwargs.get('source_vq02vq06', [[]] * batch_size)
    
    edit_infos = reward_kwargs.get('edit_info', [""] * batch_size)
    labels = reward_kwargs.get('label', [""] * batch_size) 
    
    parsed_list = []
    for i in range(batch_size):
        target_text = labels[i] if labels[i] else edit_infos[i]

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
    gemini_api_key: str = api_key, 
    **reward_kwargs
) -> List[float]:
    """
    Reward function using Gemini as a Judge for audio emotional alignment.
    """
    if not gemini_api_key:
        logger.error("Gemini API Key is missing!")
        return [0.0] * len(completion_ids)

    batch_size = len(completion_ids)
    parsed_kwargs = _parse_kwargs_for_gemini(reward_kwargs, batch_size)
    
    gemini_service = GeminiAudioService(api_key=gemini_api_key)

    results = [0.0] * batch_size
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

