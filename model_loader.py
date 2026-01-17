"""
Unified model loading utility using vLLM for high-performance inference
Supports ModelScope, HuggingFace and local path loading
"""
import os
import logging
import threading
from typing import Optional
from transformers import AutoTokenizer
from funasr_detach import AutoModel

# vLLM imports
from vllm import LLM, SamplingParams

# Global cache for downloaded models to avoid repeated downloads
_model_download_cache = {}
_download_cache_lock = threading.Lock()


class ModelSource:
    """Model source enumeration"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AUTO = "auto"


class UnifiedModelLoader:
    """Unified model loader using vLLM"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _cached_snapshot_download(self, model_path: str, source: str, **kwargs) -> str:
        """
        Cached version of snapshot_download to avoid repeated downloads
        """
        cache_key = (model_path, source, str(sorted(kwargs.items())))

        with _download_cache_lock:
            if cache_key in _model_download_cache:
                cached_path = _model_download_cache[cache_key]
                self.logger.info(f"Using cached download for {model_path} from {source}: {cached_path}")
                return cached_path

        if source == ModelSource.MODELSCOPE:
            from modelscope.hub.snapshot_download import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        elif source == ModelSource.HUGGINGFACE:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source for cached download: {source}")

        with _download_cache_lock:
            _model_download_cache[cache_key] = local_path

        self.logger.info(f"Downloaded and cached {model_path} from {source}: {local_path}")
        return local_path

    def detect_model_source(self, model_path: str) -> str:
        """Automatically detect model source"""
        if os.path.exists(model_path) or os.path.isabs(model_path):
            return ModelSource.LOCAL

        if "/" in model_path and not model_path.startswith("http"):
            if "modelscope" in model_path.lower():
                return ModelSource.MODELSCOPE
            else:
                return ModelSource.HUGGINGFACE

        return ModelSource.LOCAL

    def load_model(
        self,
        model_path: str,
        source: str = ModelSource.AUTO,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        kv_cache_dtype: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple:
        """
        Load model using vLLM for high-performance inference

        Args:
            model_path: Model path or ID
            source: Model source (auto/local/modelscope/huggingface)
            quantization: Quantization method ('awq', 'gptq', 'fp8', or None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            max_model_len: Maximum sequence length
            dtype: Data type ('float16', 'bfloat16', 'float32')
            trust_remote_code: Whether to trust remote code
            kv_cache_dtype: KV cache dtype (None, 'auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3')
            max_num_seqs: Maximum number of concurrent sequences
            max_num_batched_tokens: Maximum tokens per batch
            **kwargs: Other vLLM parameters

        Returns:
            Tuple of (llm, tokenizer, model_path)

        Example:
            >>> loader = UnifiedModelLoader()
            >>> llm, tokenizer, path = loader.load_model(
            ...     model_path="/path/to/model",
            ...     quantization="awq",
            ...     tensor_parallel_size=2
            ... )
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"🚀 Loading vLLM model from {source}: {model_path}")
        if quantization:
            self.logger.info(f"🔧 Quantization: {quantization}")

        try:
            # Resolve model path based on source
            resolved_path = model_path
            if source == ModelSource.MODELSCOPE:
                resolved_path = self._cached_snapshot_download(model_path, ModelSource.MODELSCOPE)
            elif source == ModelSource.HUGGINGFACE:
                resolved_path = self._cached_snapshot_download(model_path, ModelSource.HUGGINGFACE)

            # Build vLLM arguments
            llm_kwargs = {
                "model": resolved_path,
                "trust_remote_code": trust_remote_code,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": dtype,
                "enforce_eager": enforce_eager,
            }

            if quantization:
                llm_kwargs["quantization"] = quantization

            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len

            # Memory optimization parameters
            if kv_cache_dtype is not None:
                llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

            if max_num_seqs is not None:
                llm_kwargs["max_num_seqs"] = max_num_seqs

            if max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

            llm_kwargs.update(kwargs)

            self.logger.info(f"🔧 vLLM config: {llm_kwargs}")

            # Create vLLM LLM instance
            llm = LLM(**llm_kwargs)

            # Load tokenizer separately (needed for encoding prompts)
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                trust_remote_code=True
            )

            self.logger.info(f"✅ Successfully loaded vLLM model")
            return llm, tokenizer, resolved_path

        except Exception as e:
            self.logger.error(f"❌ Failed to load vLLM model: {e}")
            raise

    def load_funasr_model(
        self,
        repo_path: str,
        model_path: str,
        source: str = ModelSource.AUTO,
        **kwargs
    ) -> AutoModel:
        """
        Load FunASR model (for StepAudioTokenizer)

        Args:
            repo_path: Repository path
            model_path: Model path or ID
            source: Model source
            **kwargs: Other parameters

        Returns:
            FunASR AutoModel instance
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"Loading FunASR model from {source}: {model_path}")

        try:
            model_revision = kwargs.pop("model_revision", "main")

            if source == ModelSource.LOCAL:
                model_hub = "local"
            elif source == ModelSource.MODELSCOPE:
                model_hub = "ms"
            elif source == ModelSource.HUGGINGFACE:
                model_hub = "hf"
            else:
                raise ValueError(f"Unsupported model source: {source}")

            model = AutoModel(
                repo_path=repo_path,
                model=model_path,
                model_hub=model_hub,
                model_revision=model_revision,
                **kwargs
            )

            self.logger.info(f"✅ Successfully loaded FunASR model")
            return model

        except Exception as e:
            self.logger.error(f"❌ Failed to load FunASR model: {e}")
            raise


# Global instance
model_loader = UnifiedModelLoader()
