import torch
import torch.nn as nn

import os
import time

import compare_voice
import embedding_voice
import update_voice
import load_model
import Speech2Text

from pydub import AudioSegment
from playsound import playsound

def play_audio(file_path):
    print(f"재생 중: {file_path}")
    playsound(file_path, block=False)
    
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio) / 1000.0  # Convert to seconds
    
    time.sleep(duration)
    print("재생 완료")

def get_sample_voice(input_voice_num: int) -> torch.Tensor:
    voice_embed_dir: str = os.path.join('./sample_voices', f"{input_voice_num}.wav")
    play_audio(voice_embed_dir)
    input_waveform: torch.Tensor = compare_voice.load_audio(voice_embed_dir)
    return (input_waveform)


def main() -> int:
    model: nn.Module = load_model.load_driver_recognition_model()
    if model is None:
        return (1)
    input_voice_num: str = int(input("Choose Sample Voice(1 - 5) : "))
    input_waveform: torch.Tensor = get_sample_voice(input_voice_num)
    is_same, name = compare_voice.compare_voices_stdout('./voice_embeddings', input_waveform, model)
    if name is not None:
        Speech2Text.SpeechRecognition(os.path.join('./sample_voices', f"{input_voice_num}.wav"), name)
    else:
        return (1)
    
    
    return (0)

if __name__ == "__main__":
    main()