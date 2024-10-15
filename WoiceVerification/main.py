import torch
import torch.nn as nn

import os
import time

import compare_voice
import embedding_voice
import update_voice
import load_model
import Speech2Text

def get_sample_voice(input_voice_num: str) -> torch.Tensor:
    voice_embed_dir: str = os.path.join('./sample_voices', f"{input_voice_num}.wav")
    # play_audio(voice_embed_dir)
    input_waveform: torch.Tensor = compare_voice.load_audio(voice_embed_dir)
    return (input_waveform)


def main() -> int:
    model: nn.Module = load_model.load_driver_recognition_model()
    if model is None:
        return (1)
    input_voice_file: str = input("Choose Sample Voice : ")
    input_waveform: torch.Tensor = get_sample_voice(input_voice_file)
    is_same, name = compare_voice.compare_voices_stdout('./voice_embeddings', input_waveform, model)
    if name is not None:
        Speech2Text.SpeechRecognition(os.path.join('./sample_voices', f"{input_voice_file}.wav"), name)
    else:
        return (1)
    
    
    return (0)

if __name__ == "__main__":
    main()