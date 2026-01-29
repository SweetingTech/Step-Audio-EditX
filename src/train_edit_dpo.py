import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any, Dict

import transformers
from transformers import (
    HfArgumentParser, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedModel, 
    AutoConfig,
)
from trl import DPOTrainer, DPOConfig
import trl

from dataset.edit_dpodataset import EditDPODataset 
from transformers.trainer_utils import get_last_checkpoint
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset

try:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
except ImportError:
    _BaseAutoModelClass = Any

def create_edit_dpodataset(
    json_file: Union[str, List[str]], 
    max_text_length: int = 512,
    max_audio_tokens: int = 2048,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
) -> HFDataset:
    """
    Create DPO dataset.
    
     The samples yielded by the DPO dataset generator must contain the following keys:
    - 'prompt': Input prompt (text + audio context)
    - 'chosen': The preferred response (better response)
    - 'rejected': The rejected response (worse response)
    """

    core_dataset = EditDPODataset(
        json_files=json_file, 
        max_text_length=max_text_length,
        max_audio_tokens=max_audio_tokens,
        processing_class=processing_class,
        lazy=True 
    )

    def gen_dpo_data():
        for sample in core_dataset.iterate_data():
            yield sample 

    print("Creating Hugging Face DPO Dataset from generator...")
    hf_dataset = HFDataset.from_generator(gen_dpo_data)
    
    return hf_dataset


def custom_create_model_from_path(
    model_id: str, 
    architecture: Union[_BaseAutoModelClass, None] = None, 
    **kwargs
) -> PreTrainedModel:
    dtype = kwargs.get("dtype", "auto")
    if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
        pass 
    elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        raise ValueError(
            f"Invalid `dtype` passed. Expected 'auto' or torch.dtype string, got {dtype}."
        )
    
    kwargs["device_map"] = kwargs.get("device_map", "auto")

    if architecture is None:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch_name = config.architectures[0]
        
        if hasattr(transformers, arch_name):
            architecture = getattr(transformers, arch_name)
            model = architecture.from_pretrained(model_id, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                **kwargs 
            )
    else:
        model = architecture.from_pretrained(model_id, **kwargs)
        
    return model

try:
    trl.trainer.dpo_trainer.create_model_from_path = custom_create_model_from_path
except AttributeError:
    import trl.trainer.utils
    trl.trainer.utils.create_model_from_path = custom_create_model_from_path


# --- Parameter Configuration ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(default="bfloat16")

@dataclass
class DataArguments:
    data_files: List[str] = field(default_factory=list)
    max_text_length: int = field(default=512)
    max_audio_tokens: int = field(default=1024)

def main():
    # Use DPOConfig for training arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Model Initialization Arguments
    model_init_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": "eager",
    }
    training_args.model_init_kwargs = model_init_kwargs

    # 3. Create Dataset (Must contain prompt, chosen, and rejected)
    train_dataset = create_edit_dpodataset(
        json_file=data_args.data_files,
        processing_class=tokenizer
    )

    # 4. Initialize DPOTrainer
    logger.info("Initializing DPOTrainer...")
    
    trainer = DPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=None, 
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 5. Training
    last_checkpoint = None
    if training_args.resume_from_checkpoint and os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    logger.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. Save Model and Tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("DPO Training completed!")

    del trainer
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()