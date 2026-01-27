import sys
import vllm

# ================= Patch Start =================
_OriginalLLM = vllm.LLM

class TrustRemoteCodeLLM(_OriginalLLM):
    def __init__(self, *args, **kwargs):
        # 这里的修改会直接传给 vllm 的初始化
        kwargs["trust_remote_code"] = True
        print(f"🚀 [Patch生效] vLLM 初始化参数已注入: trust_remote_code=True")
        super().__init__(*args, **kwargs)

vllm.LLM = TrustRemoteCodeLLM
# ================= Patch End =================

import logging
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any
from functools import partial, update_wrapper

import transformers
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from trl import GRPOConfig, GRPOTrainer
import torch.distributed as dist
import trl

    
from utils.reward_func import cer_reward_func, sim_reward_func, emo_reward_func, mos_reward_func
from utils.reward_func_gemini import gemini_reward_func
from utils.reward_func_r1 import step_audio_r1_reward_func
from utils.reward_func_genrm import genrm_reward_func
from dataset.edit_dataset import create_edit_dataset
from transformers.trainer_utils import get_last_checkpoint
import torch
try:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
except ImportError:
    _BaseAutoModelClass = Any


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
# This ensures GRPOTrainer calls custom_create_model_from_path during initialization
trl.trainer.grpo_trainer.create_model_from_path = custom_create_model_from_path

# Avoid proxy interference with local communications
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

@dataclass
class ModelArguments:
    """
    Configuration for the model
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."}
    )

@dataclass
class DataArguments:
    """
    Configuration for data and custom rewards
    """
    data_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data files (JSONL). Can be multiple."}
    )
    max_text_length: int = field(
        default=512,
        metadata={"help": "Maximum text length for the prompt."}
    )
    max_audio_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum audio token length (completion length)."}
    )
    reward_server_ip: str = field(
        default="100.108.207.104",
        metadata={"help": "IP address for the remote reward server."}
    )
    reward_server_num: int = field(
        default=2,
        metadata={"help": "Number of reward servers."}
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["cer", "sim", "emo", "r1", "gemini", "mos"],
        metadata={"help": "List of reward functions to use. Choices: cer, sim, emo, r1, gemini"}
    )

def main():
    # 1. Parse parameters
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if training_args.get_process_log_level() == 0 else logging.WARN)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model Args: {model_args}")
    logger.info(f"Data Args: {data_args}")
    logger.info(f"Training Args: {training_args}")

    # 3. Prepare Reward Functions
    reward_ip = data_args.reward_server_ip
    n_servers = data_args.reward_server_num
    
    # Wrap Reward functions
    def make_reward_func(func, name):
        f = partial(func, server_ip=reward_ip, num_servers=n_servers)
        update_wrapper(f, func)
        f.__name__ = name 
        return f
    
    reward_registry = {
        "cer": cer_reward_func,
        "sim": sim_reward_func,
        "emo": emo_reward_func,
        "gemini": gemini_reward_func,
        "r1": step_audio_r1_reward_func,
        "mos": mos_reward_func,
        "my_genrm": genrm_reward_func,
    }
    selected_reward_funcs = []
    logger.info(f"Selected reward functions: {data_args.reward_funcs}")
    print(f"Selected reward functions: {data_args.reward_funcs}")
    for name in data_args.reward_funcs:
        if name in reward_registry:
            func_name = f"{name}_reward"
            wrapped_func = make_reward_func(reward_registry[name], func_name)
            selected_reward_funcs.append(wrapped_func)
        else:
            raise ValueError(f"Reward function '{name}' is not supported. Available: {list(reward_registry.keys())}")

    if not selected_reward_funcs:
        raise ValueError("No valid reward functions selected!")


    # reward_funcs = [
    #     make_reward_func(cer_reward_func, "cer_reward"),
    #     make_reward_func(sim_reward_func, "sim_reward"),
    #     make_reward_func(emo_reward_func, "emo_reward")
    # ]

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Ensure pad_token_id exists; required for GRPO
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Create Dataset
    training_dataset = create_edit_dataset(
        json_file=data_args.data_files,
        max_text_length=data_args.max_text_length,
        max_audio_tokens=data_args.max_audio_tokens,
        processing_class=tokenizer
    )
    logger.info(f"Created training dataset with {len(training_dataset)} samples")

    # 6. Configure Generation Kwargs
    if not training_args.use_vllm:
        training_args.generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": data_args.max_audio_tokens, 
            "temperature": training_args.temperature,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id, 
            # "top_p": 0.9, 
        }
    else:
        training_args.generation_kwargs = {
            "max_tokens": data_args.max_audio_tokens, 
            "temperature": training_args.temperature, 
            "detokenize": False,
        }
    
    training_args.max_completion_length = data_args.max_audio_tokens
    
    # Model Init Kwargs for loading the model
    model_init_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": "eager" 
    }
    training_args.model_init_kwargs = model_init_kwargs

    # 7. Initialize Trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=selected_reward_funcs,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=None,
        processing_class=tokenizer, 
    )

    # 8. Start Training
    logger.info("Starting TTS GRPO training...")
    
    # Checkpoint recovery logic
    last_checkpoint = None
    if training_args.resume_from_checkpoint and os.path.isdir(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 9. Save Model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training completed!")

    import gc
    del trainer
    gc.collect() 
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()