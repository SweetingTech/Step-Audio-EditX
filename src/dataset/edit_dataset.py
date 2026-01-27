import json
import logging
import hashlib
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from trl import apply_chat_template

AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL = """Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text: 
{prompt_text}
speaker audio tokens: 
{prompt_wav_tokens}
[speaker_end]
"""
EDIT_SYS_PROMPT = """As a highly skilled audio editing and tuning specialist, you excel in interpreting user instructions and applying precise adjustments to meet their needs. Your expertise spans a wide range of enhancement capabilities, including but not limited to:
# Emotional Enhancement
# Speaking Style Transfer
# Non-linguistic Adjustments
# Audio Tuning & Editing
Note: You will receive instructions in natural language and are expected to accurately interpret and execute the most suitable audio edits and enhancements.
"""

class EditDataset(Dataset):
    """
    Dataset class for Audio Edit training, using a conversation format.
    Supports reading and merging data from multiple JSONL files.
    """
    
    def __init__(
        self,
        json_files: Union[str, List[str]], # 修改：支持 str 或 List[str]
        max_text_length: int = 512,
        max_audio_tokens: int = 2048,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Initialize the dataset.
        Args:
            json_files: Path to a single JSONL file or a list of file paths.
            max_text_length: Length for text truncation/filtering.
            max_audio_tokens: Length for audio sequence truncation/filtering.
            processing_class: Tokenizer or processor instance.
        """
        # Unify input as a list
        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files

        self.max_text_length = max_text_length
        self.max_audio_tokens = max_audio_tokens
        
        # Initialize Tokenizer
        if processing_class is None:
            # Note: This path may need adjustment based on the actual environment
            processing_class = AutoTokenizer.from_pretrained("/data/Model_weights/Step-Audio-EditX-sft/", trust_remote_code=True)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        self.processing_class = processing_class

        # Load data from all specified files
        self.data = self.load_data()
        logging.info(f"Total samples loaded from {len(self.json_files)} files: {len(self.data)}")

    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Loads and filters training data from the self.json_files list.
        """
        all_data = []
        
        for file_path in self.json_files:
            if not os.path.exists(file_path):
                logging.warning(f"Data file not found: {file_path}, skipping.")
                continue
            
            logging.info(f"Loading data from: {file_path}")
            file_data_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # required_keys = ['source_audio', 'source_text', 'source_vq02vq06', 'target_audio', 'target_text', 'target_vq02vq06', 'edit_type', 'edit_info']
                        required_keys = ['source_audio', 'source_text', 'source_vq02vq06', 'target_text']
                        if not all(key in item for key in required_keys):
                            # logging.debug(f"Missing required fields in line {line_idx} of {file_path}")
                            continue
                        if "task_type" not in item:
                            item["task_type"] = "edit"
                        
                        if item['task_type'] == "edit":
                            processed_item = {
                                'source_text': item['source_text'],
                                'source_audio': item['source_audio'],
                                'source_vq02vq06': item['source_vq02vq06'],
                                'target_text': item['target_text'],
                                'task_type': item['task_type'],
                                'edit_type': item['edit_type'],
                                'edit_info': item['edit_info'],
                                'emotion': item['edit_info'],
                            }
                        else:
                            processed_item = {
                                'source_text': item['source_text'],
                                'source_audio': item['source_audio'],
                                'source_vq02vq06': item['source_vq02vq06'],
                                'target_text': item['target_text'],
                                'task_type': item['task_type'],
                                'edit_type': None,
                                'edit_info': item['emotion'],
                                'emotion': item['emotion'],
                            }
                        
                        all_data.append(processed_item)
                        file_data_count += 1
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON error in {file_path} line {line_idx}: {e}")
                        continue
                    except Exception as e:
                        logging.warning(f"Error processing item in {file_path} line {line_idx}: {e}")
                        continue
            
            logging.info(f"Loaded {file_data_count} valid samples from {file_path}")
            logging.info("="*200)
        
        return all_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _build_audio_edit_instruction(
        self,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None
        ) -> str:
        audio_text = audio_text.strip() if audio_text else ""
        
        safe_edit_info = str(edit_info) if edit_info is not None else ""

        if edit_type in {"emotion", "speed"}:
            if safe_edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix=f"Make the following audio more {safe_edit_info}. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "style":
            if safe_edit_info == "remove":
                instruct_prefix = f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = f"Make the following audio more {safe_edit_info} style. The text corresponding to the audio is: {audio_text}\n"
        elif edit_type == "denoise":
            instruct_prefix = f"Remove any noise from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all noise from the audio.\n"
        elif edit_type == "vad":
            instruct_prefix = f"Remove any silent portions from the given audio while preserving the voice content clearly. Ensure that the speech quality remains intact with minimal distortion, and eliminate all silence from the audio.\n"
        elif edit_type == "paralinguistic":
            instruct_prefix = f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n  The text corresponding to the audio is: {audio_text}\n"
        else:
            # 默认为空或抛出异常，视需求而定
            instruct_prefix = f"Edit the audio based on text: {audio_text}\n" 

        return instruct_prefix

    def generate_clone_voice_id(self, prompt_text, prompt_wav):
        hasher = hashlib.sha256()
        hasher.update(prompt_text.encode('utf-8'))
        wav_data = prompt_wav.cpu().numpy()
        if wav_data.size > 2000:
            audio_sample = np.concatenate([wav_data.flatten()[:1000], wav_data.flatten()[-1000:]])
        else:
            audio_sample = wav_data.flatten()
        hasher.update(audio_sample.tobytes())
        voice_hash = hasher.hexdigest()[:16]
        return f"clone_{voice_hash}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        if item['task_type'] == "edit":
            instruct_prefix = self._build_audio_edit_instruction(item['source_text'], item['edit_type'], item['edit_info'], item['source_text'])
            audio_str = self.processing_class.decode(item['source_vq02vq06'])
            messages = [
                {"role": "system", "content": EDIT_SYS_PROMPT},
                {"role": "user", "content": f"{instruct_prefix}\n{audio_str}\n"}
            ]
            pass
        else:
            prompt_wav, _ = torchaudio.load(item['source_audio'])
            prompt_speaker = self.generate_clone_voice_id(item['source_text'], prompt_wav)
            prompt_wav_tokens_str = self.processing_class.decode(item['source_vq02vq06'])
            sys_prompt = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
                speaker=prompt_speaker,
                prompt_text=item['source_text'],
                prompt_wav_tokens=prompt_wav_tokens_str
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": item['source_text']}
            ]
            pass

        return {
            'prompt': messages, 
            'source_text': item['source_text'], 
            'source_audio': item['source_audio'],
            'source_vq02vq06': item['source_vq02vq06'],
            'target_text': item['target_text'],
            'task_type': item['task_type'],
            'emotion': item['emotion'],
            'edit_type': item['edit_type'],
            'edit_info': item['edit_info'],
        }
 
    def collate_fn(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch


def create_edit_dataset(
    json_file: Union[str, List[str]], 
    max_text_length: int = 512,
    max_audio_tokens: int = 2048,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
) -> EditDataset:
    """
    Function to create an EditDataset instance.
    The json_file parameter can be a single string path or a list of string paths.
    """
    return EditDataset(
        json_files=json_file, 
        max_text_length=max_text_length,
        max_audio_tokens=max_audio_tokens,
        processing_class=processing_class,
    )




if __name__ == "__main__":
    # 1. 定义测试数据
    # 为了测试，你可以创建一个包含多个路径的列表
    data_paths = [
        "/mnt/wangyuhao/audio/Synthesis/3b-tts/editx/step_speaker/emotion/dataset_0116.jsonl",
        # "/data/wenetspeech/jsonl_data/wenetspeech_cloning_dataset.jsonl",
        # "/data/wenetspeech/jsonl_data/wenetspeech_premium_0_reformatted.jsonl"
        # "/path/to/another/dataset.jsonl" # 可选：添加第二个文件进行测试
    ]
    
    # 2. Tokenizer
    tokenizer_path = "/data/Model_weights/Step-Audio-EditX-sft/"
    if os.path.exists(tokenizer_path):
        processing_class = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        # fallback for testing if local path doesn't exist
        print(f"Warning: {tokenizer_path} not found, using dummy tokenizer for test structure.")
        # processing_class = AutoTokenizer.from_pretrained("gpt2") # example replacement
        processing_class = None

    if processing_class:
        try:
            print(f"\n--- 正在初始化数据集 (Input: {data_paths}) ---")
            
            # 3. 实例化数据集 (传入列表)
            training_dataset = create_edit_dataset(
                json_file=data_paths, # 这里传入列表
                max_text_length=512,
                max_audio_tokens=2048,
                processing_class=processing_class
            )

            # 4. 运行测试
            if len(training_dataset) > 0:
                print(f"数据集加载成功，共有 {len(training_dataset)} 条数据。")
                print("\n--- 正在测试 dataset[0] ---")
                sample = training_dataset[0]
                print(f"Sample keys: {sample.keys()}")
                print(f"Message content: {sample['prompt']}")
                print(f"Prompt content: {processing_class.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)}")
            else:
                print("警告: 数据集为空。")

        except Exception as e:
            print(f"运行失败: {e}")
            import traceback
            traceback.print_exc()