import os
import sys
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any

import transformers
from transformers import (
    HfArgumentParser, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedModel, 
    AutoConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
import trl
import torch.distributed as dist

from dataset.edit_sftdataset import EditSFTDataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import PreTrainedTokenizerBase

try:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
except ImportError:
    _BaseAutoModelClass = Any

from datasets import Dataset as HFDataset
def create_edit_sftdataset(
    json_file: Union[str, List[str]], 
    max_text_length: int = 512,
    max_audio_tokens: int = 2048,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
) -> HFDataset:
    # 1. Instantiate the core logic class (Lazy Mode)
    # This instance holds configurations and methods without loading data into RAM.
    core_dataset = EditSFTDataset(
        json_files=json_file, 
        max_text_length=max_text_length,
        max_audio_tokens=max_audio_tokens,
        processing_class=processing_class,
        lazy=True 
    )

    # 2. Define Generator function
    # This will be called by Dataset.from_generator
    def gen_edit_data():
        for sample in core_dataset.iterate_data():
            yield sample

    # 3. Create HF Dataset from generator
    # Note: 'features' can be defined to optimize storage; otherwise, HF infers them.
    print("Creating Hugging Face Dataset from generator...")
    hf_dataset = HFDataset.from_generator(gen_edit_data)
    
    return hf_dataset


def custom_create_model_from_path(
    model_id: str, 
    architecture: Union[_BaseAutoModelClass, None] = None, 
    **kwargs
) -> PreTrainedModel:
    """
    Custom model loading function supporting architectures not in the native transformers library (remote code).
    """
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
        
        # Check if config.architectures[0] exists in the transformers module
        if hasattr(transformers, arch_name):
            architecture = getattr(transformers, arch_name)
            model = architecture.from_pretrained(model_id, **kwargs)
        else:
            # If not found, instantiate using AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                **kwargs 
            )
    else:
        # Use the architecture if it is already specified via parameters
        model = architecture.from_pretrained(model_id, **kwargs)
        
    return model

# Critical Step: Replace the internal create_model_from_path used by trl
trl.trainer.sft_trainer.create_model_from_path = custom_create_model_from_path


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
    # Parse command line arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
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

    # 2. Configure model initialization arguments
    model_init_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": "eager",
    }
    training_args.model_init_kwargs = model_init_kwargs

    # 3. Create Dataset
    train_dataset = create_edit_sftdataset(
        json_file=data_args.data_files,
        processing_class=tokenizer
    )

    # 4. Initialize SFTTrainer
    # training_args.dataset_text_field = "prompt"
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 5. Training Process
    last_checkpoint = None
    if training_args.resume_from_checkpoint and os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. Save Model and Tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()