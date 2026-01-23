import os
# Force TRITON_ATTN backend for Step1 model (required for alibi_sqrt support)
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import gradio as gr
import argparse
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from datetime import datetime
import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types
from whisper_wrapper import WhisperWrapper

# Configure logging
logger = logging.getLogger(__name__)

# Save audio to temporary directory
def save_audio(audio_type, audio_data, sr, tmp_dir):
    """Save audio data to a temporary file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.debug(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


class EditxTab:
    """Audio editing and voice cloning interface tab"""

    def __init__(self, args):
        self.args = args
        self.edit_type_list = list(get_supported_edit_types().keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_auto_transcribe = getattr(args, 'enable_auto_transcribe', False)

    def history_messages_to_show(self, messages):
        """Convert message history to gradio chatbot format"""
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": gr.Audio(value=raw_audio_part, interactive=False)},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": gr.Audio(value=edit_audio_part, interactive=False)}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate cloned audio"""
        self.logger.info("Starting voice cloning process")

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state
        if edit_type != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state

        try:
            # Use common_tts_engine for cloning
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Voice cloning completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Clone failed"
                self.logger.error(error_msg)
                return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, state):
        """Generate edited audio"""
        self.logger.info("Starting audio editing process")

        # Input validation
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                self.logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                sample_rate, audio_numpy, previous_text = state["history_audio"][-1]
                temp_path = save_audio("temp", audio_numpy, sample_rate, self.args.tmp_dir)
                audio_to_edit = temp_path
                text_to_use = previous_text
                self.logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # For para-linguistic, use generated_text; otherwise use source text
            if edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, edit_type, edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                self.logger.error(error_msg)
                return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            self.logger.error(error_msg)
            return self.history_messages_to_show(state["history_messages"]) + [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }

    def register_components(self):
        """Register gradio components - maintaining exact layout from original"""
        with gr.Tab("Editx"):
            with gr.Row():
                with gr.Column():
                    self.model_input = gr.Textbox(label="Model Name", value="Step-Audio-EditX", scale=1)
                    self.prompt_text_input = gr.Textbox(label="Prompt Text", value="", scale=1)
                    self.prompt_audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        format="wav",
                        type="filepath",
                        label="Input Audio",
                    )
                    self.generated_text = gr.Textbox(label="Target Text", lines=1, max_lines=200, max_length=1000)
                with gr.Column():
                    with gr.Row():
                        self.edit_type = gr.Dropdown(label="Task", choices=self.edit_type_list, value="clone")
                        self.edit_info = gr.Dropdown(label="Sub-task", choices=[], value=None)
                    self.chat_box = gr.Chatbot(label="History", height=480)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.button_tts = gr.Button("CLONE", variant="primary")
                        self.button_edit = gr.Button("EDIT", variant="primary")
                with gr.Column():
                    self.clean_history_submit = gr.Button("Clear History", variant="primary")

            gr.Markdown("---")
            gr.Markdown("""
                **Button Description:**
                - CLONE: Synthesizes audio based on uploaded audio and text, only used for clone mode, will clear history information when used.
                - EDIT: Edits based on uploaded audio, or continues to stack edit effects based on the previous round of generated audio.
                """)
            gr.Markdown("""
                **Operation Workflow:**
                - Upload the audio to be edited on the left side and fill in the corresponding text content of the audio;
                - If the task requires modifying text content (such as clone, para-linguistic), fill in the text to be synthesized in the "target text" field. For all other tasks, keep the uploaded audio text content unchanged;
                - Select tasks and subtasks on the right side (some tasks have no subtasks, such as vad, etc.);
                - Click the "CLONE" or "EDIT" button on the left side, and audio will be generated in the dialog box on the right side.
                """)
            gr.Markdown("""
                **Para-linguistic Description:**
                - Supported tags include: [Breathing] [Laughter] [Surprise-oh] [Confirmation-en] [Uhm] [Surprise-ah] [Surprise-wa] [Sigh] [Question-ei] [Dissatisfaction-hnn]
                - Example:
                    - Fill in "target text" field: "Great, the weather is so nice today." Click the "CLONE" button to get audio.
                    - Change "target text" field to: "Great[Laughter], the weather is so nice today[Surprise-ah]." Click the "EDIT" button to get para-linguistic audio.
                """)

    def register_events(self):
        """Register event handlers"""
        # Create independent state for each session
        state = gr.State(self.init_state())

        self.button_tts.click(self.generate_clone,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])
        self.button_edit.click(self.generate_edit,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, state],
            outputs=[self.chat_box, state])

        self.clean_history_submit.click(self.clear_history, inputs=[state], outputs=[self.chat_box, state])
        self.edit_type.change(
            fn=self.update_edit_info,
            inputs=self.edit_type,
            outputs=self.edit_info,
        )

        # Add audio transcription event only if enabled
        if self.enable_auto_transcribe:
            self.prompt_audio_input.change(
                fn=self.transcribe_audio,
                inputs=[self.prompt_audio_input, self.prompt_text_input],
                outputs=self.prompt_text_input,
            )

    def update_edit_info(self, category):
        """Update sub-task dropdown based on main task selection"""
        category_items = get_supported_edit_types()
        choices = category_items.get(category, [])
        value = None if len(choices) == 0 else choices[0]
        return gr.Dropdown(label="Sub-task", choices=choices, value=value)

    def transcribe_audio(self, audio_input, current_text):
        """Transcribe audio using Whisper ASR when prompt text is empty"""
        # Only transcribe if current text is empty
        if current_text and current_text.strip():
            return current_text  # Keep existing text
        if not audio_input:
            return ""  # No audio to transcribe
        if whisper_asr is None:
            self.logger.error("Whisper ASR not initialized.")
            return ""

        try:
            # Transcribe audio
            transcribed_text = whisper_asr(audio_input)
            self.logger.info(f"Audio transcribed: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            return ""


def launch_demo(args, editx_tab):
    """Launch the gradio demo"""
    with gr.Blocks(
            theme=gr.themes.Soft(), 
            title="🎙️ Step-Audio-EditX",
            css="""
    :root {
        --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    """) as demo:
        gr.Markdown("## 🎙️ Step-Audio-EditX")
        gr.Markdown("Audio Editing and Zero-Shot Cloning using Step-Audio-EditX")

        # Register components
        editx_tab.register_components()

        # Register events
        editx_tab.register_events()

    # Launch demo
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share if hasattr(args, 'share') else False
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--server-port", type=int, default=7860, help="Demo server port.")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/gradio", help="Save path.")
    parser.add_argument("--share", action="store_true", help="Share gradio app.")

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
        default=None,
        help="Maximum number of concurrent sequences (default: 256, lower = less memory)"
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration (default: max_model_len, lower = less activation memory)"
    )
    parser.add_argument(
        "--enable-auto-transcribe",
        action="store_true",
        help="Enable automatic audio transcription when uploading audio files (default: disabled)"
    )
    
    # CosyVoice vocoder parameters
    parser.add_argument(
        "--cosyvoice-dtype",
        type=str,
        default="float32",
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

    args = parser.parse_args()

    # Map string arguments to actual types
    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    # Determine tokenizer path
    if args.tokenizer_path:
        tokenizer_base_path = args.tokenizer_path
    else:
        # Try sibling directory first, then subdirectory
        parent_dir = os.path.dirname(args.model_path)
        sibling_tokenizer = os.path.join(parent_dir, "Step-Audio-Tokenizer")
        sub_tokenizer = os.path.join(args.model_path, "Step-Audio-Tokenizer")
        
        # For HuggingFace/ModelScope paths (e.g., "stepfun-ai/xxx"), always use sibling path
        # since os.path.exists() only works for local filesystem
        if model_source in [ModelSource.HUGGINGFACE, ModelSource.MODELSCOPE]:
            tokenizer_base_path = sibling_tokenizer
        elif os.path.exists(sibling_tokenizer):
            tokenizer_base_path = sibling_tokenizer
        else:
            tokenizer_base_path = sub_tokenizer

    logger.info(f"🚀 Loading models with vLLM backend")
    logger.info(f"   - Model source: {args.model_source}")
    logger.info(f"   - Model path: {args.model_path}")
    logger.info(f"   - Tokenizer path: {tokenizer_base_path}")
    logger.info(f"   - dtype: {args.dtype}")
    logger.info(f"   - Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"   - GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"   - Max model length: {args.max_model_len}")
    if args.tts_model_id:
        logger.info(f"   - TTS model ID: {args.tts_model_id}")
    if args.quantization:
        logger.info(f"   - Quantization: {args.quantization}")

    # Initialize models
    try:
        # Load StepAudioTokenizer (uses default funasr_model_id from tokenizer.py)
        encoder = StepAudioTokenizer(
            tokenizer_base_path,
            model_source=model_source
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")
        
        # Initialize TTS engine with vLLM
        common_tts_engine = StepAudioTTS(
            args.model_path,
            encoder,
            model_source=model_source,
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
        logger.info("✓ StepCommonAudioTTS loaded successfully")
        
        if args.enable_auto_transcribe:
            whisper_asr = WhisperWrapper()
            logger.info("✓ Automatic audio transcription enabled")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)

    # Create EditxTab instance
    editx_tab = EditxTab(args)

    # Launch demo
    launch_demo(args, editx_tab)
