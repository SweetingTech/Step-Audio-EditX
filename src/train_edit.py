import sys
import vllm

# ================= 补丁开始 =================
# 1. 保存原版 LLM
_OriginalLLM = vllm.LLM

# 2. 定义魔改版 LLM，强制注入 trust_remote_code=True
class TrustRemoteCodeLLM(_OriginalLLM):
    def __init__(self, *args, **kwargs):
        # 这里的修改会直接传给 vllm 的初始化
        kwargs["trust_remote_code"] = True
        print(f"🚀 [Patch生效] vLLM 初始化参数已注入: trust_remote_code=True")
        super().__init__(*args, **kwargs)

# 3. 替换 vllm 模块中的 LLM 类
vllm.LLM = TrustRemoteCodeLLM
# ================= 补丁结束 =================

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
    自定义的模型加载函数，支持不在 transformers 原生库中的架构（remote code）。
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
        
        # 判断 config.architectures[0] 是否在 transformers 模块中
        if hasattr(transformers, arch_name):
            architecture = getattr(transformers, arch_name)
            model = architecture.from_pretrained(model_id, **kwargs)
        else:
            # 如果不在，则使用 AutoModelForCausalLM 实例化 
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                **kwargs 
            )
    else:
        # 如果 architecture 已经通过参数指定了，则直接使用
        model = architecture.from_pretrained(model_id, **kwargs)
        
    return model

# 关键步骤：替换 trl 内部使用的 create_model_from_path 函数
# 这样 GRPOTrainer 初始化时就会调用上面的 custom_create_model_from_path
trl.trainer.grpo_trainer.create_model_from_path = custom_create_model_from_path


# 避免代理干扰本地通信
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

@dataclass
class ModelArguments:
    """
    模型相关的配置
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
    数据和自定义Reward相关的配置
    """
    data_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data files (JSONL). Can be multiple."}
    )
    max_text_length: int = field(
        default=512,
        metadata={"help": "Maximum text length for the prompt."}
    )
    # 注意：max_completion_length 在 GRPOConfig 中已有，但为了数据处理方便，这里保留一个别名或复用
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
    # 1. 统一解析参数
    # HfArgumentParser 可以同时处理多个 dataclass，并将命令行参数自动分配给它们
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    
    # 如果运行参数在 sys.argv 中，直接解析
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 支持 json 配置文件
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 设置日志
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

    # 3. 准备 Reward Functions
    # 使用参数中的 IP，而不是硬编码
    reward_ip = data_args.reward_server_ip
    n_servers = data_args.reward_server_num
    
    # 封装 Reward 函数
    def make_reward_func(func, name):
        f = partial(func, server_ip=reward_ip, num_servers=n_servers)
        update_wrapper(f, func)
        f.__name__ = name # 确保 wandb 记录时名字正确
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
            # 自动生成 reward 的名字，例如 cer_reward
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

    # 4. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # 确保 pad_token_id 存在，GRPO 需要
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. 创建数据集
    training_dataset = create_edit_dataset(
        json_file=data_args.data_files,
        max_text_length=data_args.max_text_length,
        max_audio_tokens=data_args.max_audio_tokens,
        processing_class=tokenizer
    )
    logger.info(f"Created training dataset with {len(training_dataset)} samples")

    # 6. 配置 Generation Kwargs
    # 这些参数用于 GRPO 采样阶段
    # 可以在这里设置默认值，也可以根据需要调整
    if not training_args.use_vllm:
        training_args.generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": data_args.max_audio_tokens, # 保持与数据一致
            "temperature": training_args.temperature, # 复用 training_args 中的 temperature
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id, 
            # "top_p": 0.9, # 可选
        }
    else:
        training_args.generation_kwargs = {
            "max_tokens": data_args.max_audio_tokens, # 保持与数据一致
            "temperature": training_args.temperature, # 复用 training_args 中的 temperature
            "detokenize": False,
        }
    
    # 强制覆盖 max_completion_length 以匹配数据处理逻辑
    training_args.max_completion_length = data_args.max_audio_tokens
    
    # Model Init Kwargs 用于加载模型
    model_init_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": "eager" # 根据实际情况选择 flash_attention_2 或 eager
    }
    training_args.model_init_kwargs = model_init_kwargs

    # 7. 初始化 Trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=selected_reward_funcs,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=None,
        processing_class=tokenizer, # 传入 tokenizer 以处理 padding
        # model_init_kwargs=model_init_kwargs # GRPOTrainer 最新版支持直接传 kwargs，如果报错请在 model 参数直接加载模型对象
    )

    # 8. 开始训练
    logger.info("Starting TTS GRPO training...")
    
    # Checkpoint 恢复逻辑
    last_checkpoint = None
    if training_args.resume_from_checkpoint and os.path.isdir(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 9. 保存模型
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training completed!")

    # --- 新增：尝试手动删除 trainer 以触发清理 ---
    import gc
    del trainer
    gc.collect() 
    torch.cuda.empty_cache()
    # ----------------------------------------

    # 清理分布式进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()