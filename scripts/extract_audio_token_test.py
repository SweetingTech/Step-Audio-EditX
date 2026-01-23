import json
import os
import sys  
import torchaudio

## import 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)  
if project_root not in sys.path:                          # 将项目根目录添加到sys.path
    sys.path.insert(0, project_root)
from tokenizer import StepAudioTokenizer

def test():
    audio_path = "assets/test.wav"
    tokenizer_path = "/mnt/wby-jfs/models/open_source/Step-Audio-EditX/Step-Audio-Tokenizer/"
    tokenizer = StepAudioTokenizer(
        tokenizer_path,
        model_source="local"
    )
    audio, sr = torchaudio.load(audio_path)
    audio_token, _, _ = tokenizer.wav2token(audio, sr)
    print(audio_token)

if __name__ == "__main__":
    test()