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

# --- Logging Configuration ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. Base Configuration & Constants
# ==========================================

# Flow Server port configuration (usually remains constant)
PORT_BASE_FLOW = 8080

# Default evaluation prompt template (can be overridden in subclasses)
DEFAULT_EVAL_PROMPT = """You are a professional audio emotion verification expert.

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

# ==========================================
# 2. Abstract Base Class: Generative RM Interface
# ==========================================

class BaseGenerativeRM(ABC):
    """
    Base class for all Generative Reward Models.
    Users must inherit from this class and implement the `call_model` method.
    """
    def __init__(self, prompt_template: str = DEFAULT_EVAL_PROMPT):
        self.prompt_template = prompt_template

    def format_prompt(self, target_instruction: str) -> str:
        return self.prompt_template.format(target_instruction=target_instruction)

    @abstractmethod
    def call_model(self, prompt: str, audio_base64: str) -> Optional[Union[bool, float]]:
        """
        [Abstract Method - Implementation Required]
        Calls a specific LLM/VLM API for inference.
        
        Args:
            prompt: The constructed text prompt.
            audio_base64: The base64 encoded string of the audio file (no header).
            
        Returns:
            float (0.0 - 1.0) or bool (True/False).
            Returns None if the call fails.
        """
        pass

    def parse_response(self, response_text: str) -> float:
        """
        General parsing logic to extract scores from a JSON string.
        Supports {"is_match": true} or {"score": 0.8}.
        """
        if not response_text:
            return 0.0
            
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(clean_text)
            
            if "score" in data:
                return float(data["score"])
            
            res = data.get("is_match")
            if isinstance(res, bool):
                return 1.0 if res else 0.0
            if str(res).lower() == "true": return 1.0
            if str(res).lower() == "false": return 0.0
            
        except json.JSONDecodeError:
            if re.search(r'"is_match"\s*:\s*true', clean_text.lower()): return 1.0
            if re.search(r'"is_match"\s*:\s*false', clean_text.lower()): return 0.0
            
        logger.warning(f"Failed to parse response: {clean_text[:100]}...")
        return 0.0

    def save_temp_audio(self, audio_base64: str, suffix=".wav") -> str:
        """
        [Utility Function] If the model only accepts file paths, use this to save base64 as a temporary file.
        Usage:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(base64.b64decode(audio_base64))
                tmp.flush()
                # call your model with tmp.name
        """
        pass 

# ==========================================
# 3. [User Implementation Area] Custom GenRM
# ==========================================

class CustomGenerativeRM(BaseGenerativeRM):
    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = kwargs.get("api_url", "")

    def call_model(self, prompt: str, audio_base64: str) -> Optional[float]:
        """
        TODO: Implement specific API call logic here (e.g., OpenAI / Gemini / Claude / Local vLLM).
        """
        
        # --- Example: Pseudo-code structure ---
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Construct Payload (Modify according to actual model API format)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        # Assuming API supports direct base64 transmission 
                        # If URL or file upload is required, handle that here.
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
            # # Execute request (Commented out to prevent errors during dry runs)
            # response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            # response_json = response.json()
            # content = response_json['choices'][0]['message']['content']
            
            # --- Mock return for testing ---
            # Remove this line in production and return self.parse_response(content)
            import random
            return 1.0 if random.random() > 0.5 else 0.0
            
        except Exception as e:
            logger.error(f"Model Inference Error: {e}")
            return None

# ==========================================
# 4. Flow Server Interaction
# ==========================================

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
    """Requests audio synthesis from the Flow Server"""
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
# 5. Core Processing Pipeline
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
    Single sample processing pipeline:
    Flow synthesis -> Construct Prompt -> GenRM Scoring -> Parse Score
    """
    proxies = {"http": None, "https": None}
    
    # 1. Extract target instruction (supports 'label' or 'edit_info' fields)
    target_instruction = kwargs.get('label') or kwargs.get('edit_info') or "unknown"
    
    # 2. Synthesize Audio
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

    # 3. Prepare Prompt
    prompt = genrm_model.format_prompt(target_instruction=str(target_instruction))

    # 4. Model Inference
    result = genrm_model.call_model(prompt, audio_b64)

    # 5. Return results
    if result is None:
        return 0.0 
    
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    return float(result)

# ==========================================
# 6. Reward Function Entry Point
# ==========================================

def _parse_batch_kwargs(reward_kwargs: Dict, batch_size: int) -> List[Dict]:
    """Converts batched kwargs into a list of dictionaries"""
    keys = ['source_audio', 'source_vq02vq06', 'edit_info', 'label']
    parsed_list = []
    
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
    genrm_api_key: str = os.getenv("GENRM_API_KEY", ""), 
    genrm_api_url: str = "",
    genrm_model_name: str = "gpt-4o",
    **reward_kwargs
) -> List[float]:
    """
    General Entry Point for the Generative Reward Function
    """
    batch_size = len(completion_ids)
    parsed_kwargs = _parse_batch_kwargs(reward_kwargs, batch_size)
    
    # Initialize the user-defined GenRM
    # Parameters can be adjusted based on CustomGenerativeRM's __init__ method
    genrm_model = CustomGenerativeRM(
        api_key=genrm_api_key, 
        api_url=genrm_api_url, 
        model_name=genrm_model_name
    )

    results = [0.0] * batch_size
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