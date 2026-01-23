import os
import re
import torch
from datetime import datetime
from typing import List, Union
import logging
from transformers import LlamaTokenizer
from sentencepiece import SentencePieceProcessor


def tts_accuracy_reward(completions: List[str], target_tokens: List[Union[List[int], torch.Tensor]], **kwargs) -> List[float]:
    """
    Reward function for TTS model that checks if generated audio tokens exactly match target tokens.
    Both completions and target_tokens are compared in their decoded audio token format.
    
    Args:
        completions: List of generated completion strings (containing audio tokens like <audio_123>)
        target_tokens: List of target audio token sequences (can be List[int] or torch.Tensor)
        
    Returns:
        List of rewards (1.0 for exact match, 0.0 otherwise)
    """
    if len(completions) != len(target_tokens):
        raise ValueError(f"Length mismatch: completions {len(completions)}, targets {len(target_tokens)}")
    
    # Initialize tokenizer for decoding target tokens
    model_path = "/mnt/gpfs/tianfei/tmp/ckpt_form_yanchao/step1_3b_vq0206"  # Adjust path as needed
    try:
        # tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer = SentencePieceProcessor(model_path + "tokenizer.model")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return [0.0] * len(completions)
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for i, (completion, target) in enumerate(zip(completions, target_tokens)):
        reward = 0.0
        
        try:
            # Normalize target tokens to list of integers
            if isinstance(target, torch.Tensor):
                target_list = target.cpu().tolist()
            else:
                target_list = target
            
            # Decode target tokens to audio token format (like <audio_123>)
            target_decoded = tokenizer.decode(target_list)
            
            # Clean both strings
            completion_clean = completion.strip()
            target_clean = target_decoded.strip()
            
            print(f'Completion: {completion_clean}')
            # print(f'Target: {target_clean}')
            print(f"Completion: {tokenizer(completion_clean)['input_ids']}")
            print(f'Target: {target_list}')
            # Direct string comparison of decoded audio tokens
            if completion_clean == target_clean:
                reward = 1.0
            else:
                reward = 0.0
                
        except Exception as e:
            logging.warning(f"Error processing tokens at index {i}: {e}")
            reward = 0.0
        
        rewards.append(reward)
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} TTS Accuracy reward: {reward} -------------\n")
                f.write(f"Completion: {completion}\n")
                f.write(f"Target: {target}\n")
    
    return rewards


def _clean_token_string(token_string: str, special_tokens: List[str] = None) -> str:
    """
    Remove special tokens from audio token string.
    
    Args:
        token_string: Space-separated audio token string
        special_tokens: List of special token strings to remove
    
    Returns:
        Cleaned token string
    """
    if special_tokens is None:
        # Default special tokens (BOS, EOS, PAD) - adjust based on your model
        special_tokens = ['70000', '70001', '70002', '1', '2', '0']  # audio_pad, audio_bos, audio_eos, bos, eos, pad
    
    # Split into individual tokens
    tokens = token_string.split()
    
    # Filter out special tokens
    cleaned_tokens = [token for token in tokens if token not in special_tokens]
    
    # Join back into string
    return ' '.join(cleaned_tokens)


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion has audio tokens in correct format.
    """
    rewards = []
    
    for completion in completions:
        reward = 0.0
        try:
            # Check if completion contains valid audio token format like <audio_123>
            completion_clean = completion.strip()
            if completion_clean:
                # Check if contains audio tokens in the correct format
                audio_pattern = r'<audio_\d+>'
                audio_tokens = re.findall(audio_pattern, completion_clean)
                if audio_tokens:  # Contains at least one audio token
                    reward = 1.0
        except Exception:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

import random
from typing import List

def random_reward(completions: List[str], **kwargs) -> List[float]:
    """
    随机奖励函数：针对每个生成结果，以 50% 的概率给出 1.0 或 0.0。
    该函数通常用于测试奖励系统的鲁棒性或作为 RL 训练中的基准。
    
    Args:
        completions: 模型生成的字符串列表
        **kwargs: 接收其他可能传入的参数（如 prompts, target_tokens 等），但在此函数中不使用
        
    Returns:
        List[float]: 随机的奖励值列表（0.0 或 1.0）
    """
    # 为每个 completion 生成一个随机奖励
    # random.choice 从 [0.0, 1.0] 中等概率随机选择一个
    rewards = [float(random.choice([0, 1])) for _ in completions]
    
    return rewards


