import torchaudio
import argparse
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS

def get_args():
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to Step-Audio-Tokenizer directory. If not specified, auto-detects sibling directory"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "fp8"],
        help="Enable quantization for vLLM: awq, gptq, or fp8"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization ratio 0.0-1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=3072,
        help="Maximum model sequence length, affects KV cache size (default: 8192)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for model (default: bfloat16)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graphs to save ~0.5GB GPU memory (slower inference)"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        help="KV cache data type: fp8_e5m2 reduces KV cache memory by ~50%% (default: auto, uses model dtype)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of concurrent sequences (default: 256, lower = less memory)"
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration (default: max_model_len, lower = less activation memory)"
    )
    
    # CosyVoice vocoder parameters
    parser.add_argument(
        "--cosyvoice-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="CosyVoice vocoder dtype: bfloat16 reduces memory by ~50%% (default: float32)"
    )
    parser.add_argument(
        "--no-cosyvoice-cuda-graph",
        dest="cosyvoice_cuda_graph",
        action="store_false",
        help="Disable CUDA Graph for CosyVoice vocoder (saves memory but slower)"
    )
    parser.set_defaults(cosyvoice_cuda_graph=True)

    # clone or edit parameters
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="prompt text for editing or cloning"
    )

    parser.add_argument(
        "--prompt-audio",
        type=str,
        default="",
        help="prompt audio for editing or cloning"
    )

    parser.add_argument(
        "--edit-type",
        type=str,
        choices=["clone", "emotion", "style", "vad", "denoise", "paralinguistic", "speed"],
        default="clone",
        help="Edit type"
    )

    parser.add_argument(
        "--edit-info",
        type=str,
        choices=[
            # default
            '',
            # emotion
            'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
            'empathy', 'embarrass', 'fear', 'surprised', 'excited',
            'depressed', 'coldness', 'admiration', 'remove',
            # style
            'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
            'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
            'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
            'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
            'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply', 'loudly',
            'remove', 'exaggerated',
            # speed
            'faster', 'slower', 'more faster', 'more slower'
        ],
        default="",
        help="Edit info/sub-type"
    )
    parser.add_argument(    
        "--generated-text",
        type=str,
        default="",
        help="Generated text for cloning or editing(paralinguistic)"
    )

    parser.add_argument("--output-dir", type=str, default="./output_dir", help="Save path.")

    args = parser.parse_args()

    return args

def load_model(args) -> StepAudioTTS:
    step_audio_editx_model_path = args.model_path
    step_audio_tokenizer_path = args.tokenizer_path

    step_audio_tokenizer = StepAudioTokenizer(
        step_audio_tokenizer_path,
        model_source=args.model_source
    )
    step_audio_editx = StepAudioTTS(
        step_audio_editx_model_path,
        step_audio_tokenizer,
        model_source=args.model_source,
        tts_model_id=args.tts_model_id,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        cosyvoice_dtype=args.cosyvoice_dtype,
        cosyvoice_cuda_graph=args.cosyvoice_cuda_graph
    )
    return step_audio_editx

def infer_test():
    model = load_model(args=get_args())
    prompt_audio_path = "assets/test.wav"
    prompt_audio_text = "这是一条测试音频，尝试各种功能是否正常运行。"

    ## clone
    output_audio, output_sr = model.clone(
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_audio_text,
        target_text="你好，这是一个测试。"
    )
    torchaudio.save("clone_output.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] clone_output.wav")

    ## emotion
    output_audio, output_sr  = model.edit(
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_audio_text,
        edit_type="emotion",
        edit_info="happy",
    )
    torchaudio.save("emotion_output.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] emotion_output.wav")

    ## style
    output_audio, output_sr  = model.edit(
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_audio_text,
        edit_type="style",
        edit_info="older",
    )
    torchaudio.save("style_output.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] style_output.wav")

    ## paralinguistic
    output_audio, output_sr  = model.edit(
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_audio_text,
        edit_type="paralinguistic",
        target_text="这是一条测试音频，[Laughter]尝试各种功能是否正常运行。",
    )
    torchaudio.save("paralinguistic_output.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] paralinguistic_output.wav")

    ## speed
    output_audio, output_sr  = model.edit(
        prompt_wav_path=prompt_audio_path,
        prompt_text=prompt_audio_text,
        edit_type="speed",
        edit_info="more faster",
    )
    torchaudio.save("speed_output.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] speed_output.wav")

    ## speed iter_2
    output_audio, output_sr  = model.edit(
        prompt_wav_path="./speed_output.wav",
        prompt_text=prompt_audio_text,
        edit_type="speed",
        edit_info="more faster",
    )
    torchaudio.save("speed_output2.wav", output_audio.cpu(), output_sr)
    print(f"[Saved] speed_output2.wav")

def infer():
    args = args=get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args)
    if args.edit_type == "clone":
        save_path = f"{args.output_dir}/test_clone.wav"
        output_audio, output_sr  = model.clone(
            prompt_wav_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            target_text=args.generated_text,
        )
        torchaudio.save(save_path, output_audio.cpu(), output_sr)
        print(f"[Saved] {save_path}")
    else:
        save_path = f"{args.output_dir}/test_{args.edit_type}.wav"
        if args.edit_info != "":
            save_path = f"{args.output_dir}/test_{args.edit_type}_{args.edit_info}.wav"
        output_audio, output_sr  = model.edit(
            prompt_wav_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            target_text=args.generated_text,
            edit_type=args.edit_type,
            edit_info=args.edit_info,
        )
        torchaudio.save(save_path, output_audio.cpu(), output_sr)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    infer()
    # infer_test()