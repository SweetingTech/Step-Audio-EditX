import json
import logging
import hashlib
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset as TorchDataset 
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from datasets import Dataset as HFDataset, Features, Value, Sequence 

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

class EditDPODataset(TorchDataset):
    
    def __init__(
        self,
        json_files: Union[str, List[str]], 
        max_text_length: int = 512,
        max_audio_tokens: int = 2048,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        lazy: bool = False 
    ):
        """
        Initialize the dataset.
        Args:
            json_files: A single JSONL file path or a list of file paths.
            max_text_length: Text truncation/filtering length.
            max_audio_tokens: Audio sequence truncation/filtering length.
            lazy: If True, data is not loaded into memory during init, but iterated via generator.
        """
        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files

        self.max_text_length = max_text_length
        self.max_audio_tokens = max_audio_tokens
        self.lazy = lazy
        
        if processing_class is None:
            # Note: This path may need adjustment based on the actual environment
            processing_class = AutoTokenizer.from_pretrained("/data/Model_weights/Step-Audio-EditX-sft/", trust_remote_code=True)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        self.processing_class = processing_class

        if not self.lazy:
            self.data = self.load_data()
            logging.info(f"Total samples loaded from {len(self.json_files)} files: {len(self.data)}")
        else:
            self.data = []
            logging.info(f"Lazy mode enabled. Will iterate over {len(self.json_files)} files during generation.")

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

    def process_item_raw(self, item: Dict[str, Any]) -> Dict[str, Any]:
        target_vq = item.get('target_vq02vq06', None)
        if item.get("task_type", "edit") == "edit":
            processed_base = {
                'source_text': item['source_text'],
                'source_audio': item['source_audio'],
                'source_vq02vq06': item['source_vq02vq06'],
                'target_text': item['target_text'],
                'target_vq02vq06': target_vq,
                'task_type': item.get("task_type", "edit"),
                'edit_type': item['edit_type'],
                'edit_info': item['edit_info'],
                'emotion': item['edit_info'],
            }
        else:
            processed_base = {
                'source_text': item['source_text'],
                'source_audio': item['source_audio'],
                'source_vq02vq06': item['source_vq02vq06'],
                'target_text': item['target_text'],
                'target_vq02vq06': target_vq,
                'task_type': item.get("task_type", "edit"),
                'edit_type': None,
                'edit_info': item['emotion'],
                'emotion': item['emotion'],
            }

        chosen_audio_str = self.processing_class.decode(processed_base['target_vq02vq06'])
        rejected_audio_str = self.processing_class.decode(processed_base['source_vq02vq06'])
        
        if processed_base['task_type'] == "edit":
            instruct_prefix = self._build_audio_edit_instruction(
                processed_base['source_text'], processed_base['edit_type'], processed_base['edit_info'], processed_base['source_text']
            )
            audio_str = self.processing_class.decode(processed_base['source_vq02vq06'])
            
            prompt = [
                {"role": "system", "content": EDIT_SYS_PROMPT},
                {"role": "user", "content": f"{instruct_prefix}\n{audio_str}\n"},
            ]
            chosen = [
                {"role": "assistant", "content": f"{chosen_audio_str}"},
            ]
            rejected = [
                {"role": "assistant", "content": f"{rejected_audio_str}"}
            ]
        else:
            prompt_wav, _ = torchaudio.load(processed_base['source_audio'])
            prompt_speaker = self.generate_clone_voice_id(processed_base['source_text'], prompt_wav)
            prompt_wav_tokens_str = self.processing_class.decode(processed_base['source_vq02vq06'])
            sys_prompt = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
                speaker=prompt_speaker,
                prompt_text=processed_base['source_text'],
                prompt_wav_tokens=prompt_wav_tokens_str
            )
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": processed_base['source_text']},
            ]
            chosen = [
                {"role": "assistant", "content": f"{chosen_audio_str}"},
            ]
            rejected = [
                {"role": "assistant", "content": f"{rejected_audio_str}"}
            ]


        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        

    def iterate_data(self):
        required_keys = ['source_audio', 'source_text', 'source_vq02vq06', 'target_text']
        
        for file_path in self.json_files:
            if not os.path.exists(file_path):
                logging.warning(f"Data file not found: {file_path}, skipping.")
                continue
            
            logging.info(f"Stream loading data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if not all(key in item for key in required_keys):
                            continue
                        
                        processed_sample = self.process_item_raw(item)
                        yield processed_sample
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logging.warning(f"Error processing item in {file_path} line {line_idx}: {e}")
                        continue

    def load_data(self) -> List[Dict[str, Any]]:
        all_data = []
        for sample in self.iterate_data():
            all_data.append(sample)
        return all_data

    def __len__(self) -> int:
        if self.lazy:
             return 0 
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.lazy:
            raise NotImplementedError("Lazy mode does not support random access via __getitem__")
        return self.data[idx] 


def create_edit_dpodataset(
    json_file: Union[str, List[str]], 
    max_text_length: int = 512,
    max_audio_tokens: int = 2048,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
) -> HFDataset:
    
    core_dataset = EditDPODataset(
        json_files=json_file, 
        max_text_length=max_text_length,
        max_audio_tokens=max_audio_tokens,
        processing_class=processing_class,
        lazy=True 
    )

    def gen_edit_data():
        for sample in core_dataset.iterate_data():
            yield sample

    print("Creating Hugging Face Dataset from generator...")
    hf_dataset = HFDataset.from_generator(gen_edit_data)
    
    return hf_dataset
