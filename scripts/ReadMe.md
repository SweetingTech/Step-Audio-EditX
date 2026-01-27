# Step-audio-editx GRPO Training Framework

This project is a GRPO training framework implemented for editx, based on the TRL library.

## Quick Start

### 1. Environment Preparation

Ensure the following dependencies are installed:
```bash
cd Step-Audio-EditX
uv sync --refresh
source .venv/bin/activate
```
### 2. Data Preparation

Data should be in JSONL format, with each line containing the following fields. For audio token extraction, please refer to `scripts/extract_audio_token_test.py`
```json
{
    "source_audio": "Path to the audio to be edited",
    "source_text": "Text corresponding to source_audio",
    "source_vq02vq06": "Audio token sequence of source_audio",
    "target_text": "Target text for the edited audio",
    "task_type": "edit",
    "edit_type": "emotion",
    "edit_info": "Specific emotion category",
}
```

### 3. Model Configuration
Before starting the training, check your model folder. The directory should contain full model weights and configuration files with the following structure:
```
Step-Audio-EditX/
├── CosyVoice-300M-25Hz
│   └── ...
├── config.json
├── configuration_step1.py
├── configuration.json
├── model-00001.safetensors
├── model.safetensors.index.json
├── modeling_step1.py
├── tokenizer_config.json
└── tokenizer.model
```

Additionally, you must manually modify the configuration file:

- modeling_step1.py: Add `self.gradient_checkpointing = False` around line 340.
```python
    ...
    def __init__(self, config):
        super().__init__(config)
        self.model = Step1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
    ...
```


### 4 Deploy Flow-matching Inference Service

This project decouples the Flow-matching component from the training logic. You must deploy the Flow-matching inference service before launching the training script.

**Step 1: Configure Model Path**

In `src/utils/flow_server.py` (line 15), update the `FLOW_PATH` variable to point to your local **CosyVoice-300M-25Hz** model path.

```python
# src/utils/flow_server.py
FLOW_PATH = "/path/to/your/pretrained_models/CosyVoice-300M-25Hz"
```

**Step 2: Start the Service**

Run the following commands to start the Flow-matching Server:

```bash
cd ./src/utils/
bash run_server.sh
```

### 5. Define Reward Functions

#### 5.1 Custom Reward Function

You can quickly integrate any multimodal LLM with audio understanding capabilities as a reward model based on `reward_func_genrm.py`.


**Step 1: Implement the API Interface**
Open `src/utils/reward_func_genrm.py` and locate the `CustomGenerativeRM` class. You only need to implement the `call_model` method to handle the API request to your target model.

**Step 2: Register and Enable the Reward**
Register your function in the training script `src/train_edit.py`.
```python
# src/train_edit.py
from utils.reward_func_genrm import genrm_reward_func

def main():
    # ...
    reward_registry = {
        # New entry: The key is the name used in the launch script
        "my_genrm": genrm_reward_func, 
    }
```

In your launch script, add the registered name to `REWARD_FUNCS`:

```bash
# ./scripts/run_edit_grpo.sh
REWARD_FUNCS="my_genrm"
```

#### 5.2 Using Gemini as a Reward Model
A Gemini-based reward function is implemented in `src/utils/reward_func_gemini.py`. To use it:

**Step 1: Configure Environment Variable**
Set your Google API Key in the terminal:
```bash
export API_KEY="YOUR_GEMINI_API_KEY"
```

**Step 2: Enable in Training Script**
Set the `REWARD_FUNCS` parameter to "gemini":
```bash
# ./scripts/run_edit_grpo.sh
REWARD_FUNCS="gemini"
```


### 6. Launching Training

Once the Flow-matching service is running (port listening without errors) and reward functions are defined, you can start training.

#### 6.1 Training Scripts
The project provides two scripts:

- `run_edit_grpo.sh`: Uses standard Hugging Face inference mode.

- `run_edit_grpo_vllm.sh`: Uses vLLM for inference sampling, providing faster speeds and better VRAM efficiency.


#### 6.2 Modifying Configuration
Update the parameters in the `./scripts/` directory before execution:
```bash
#!/bin/bash

# Model and Data Paths
MODEL_PATH="{EDITX_PATH}"                # Path to Step-audio-editx model
DATA_FILES=(
    "{TRAINING_INDEX_FILE}"               # Path to training data index (JSONL)
)

# Output and Logging
OUTPUT_DIR="{YOUR_PATH_TO_SAVE_CHECKPOINT}"
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"
CONFIG_PATH="./config/train_config/accelerate_configs/deepspeed_zero2.yaml"

# Reward and Server Config
REWARD_FUNCS="my_genrm"                   # Enabled Reward Functions
SERVER_IP="127.0.0.1"                     # IP of the Flow-matching service
...

```


### 7. Key Training Parameters

| Parameter | Default/Example | Description |
| :---- | :--- | :--- |
| `--num_generations` | `8` | **Group Size**. The number of samples generated per prompt (Core GRPO parameter). |
| `--per_device_train_batch_size` | `2` | Number of input prompts per GPU. |
| `--gradient_accumulation_steps` | `2` | Steps for gradient accumulation to simulate larger batches. |
| `--max_audio_tokens` | `1024` | Maximum length for generated audio tokens. |
| `--reward_server_ip` | `{SERVER_IP}` | IP of the Flow-matching service (usually `127.0.0.1`). |
| `--reward_server_num` | `2` | Number of concurrent Flow-matching server instances (must match `NUM_SERVERS` in `run_server.sh`). |
| `--reward_funcs` | `...` | List of enabled reward functions. |
| `--use_vllm` | `false` | Enable vLLM acceleration for faster inference sampling. |
| `--vllm_mode` | `"colocate"` | Deployment mode for vLLM. |
| `--vllm_gpu_memory_utilization` | `0.3` | Percentage of VRAM (0.0-1.0) allocated for vLLM. |



