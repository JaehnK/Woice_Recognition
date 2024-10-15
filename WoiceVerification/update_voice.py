import numpy as np
import torch
import torch.nn as nn
import torchaudio
import os
from typing import List
from pathlib import Path

from load_model import load_driver_recognition_model

# ReDimNet 모델 로드 (이전 코드에서 정의한 대로)
# model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)

def load_audio(file_path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform

def generate_embedding(model: torch.nn.Module, audio: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embedding = model(audio)
    return embedding.cpu().numpy().flatten()

def update_embedding(current_embedding: np.ndarray, new_embedding: np.ndarray, weight: float = 0.7) -> np.ndarray:
    """
    현재 임베딩과 새 임베딩을 결합하여 업데이트된 임베딩을 반환합니다.
    :param current_embedding: 현재 저장된 임베딩
    :param new_embedding: 새로운 음성 데이터의 임베딩
    :param weight: 현재 임베딩의 가중치 (0 ~ 1 사이의 값)
    :return: 업데이트된 임베딩
    """
    return weight * current_embedding + (1 - weight) * new_embedding

def update_voice_profile(stored_embedding_path: str, new_audio_files: List[str], 
                        output_path: str, model: nn.Module, weight: float = 0.7):
    # 저장된 임베딩 로드
    current_embedding = np.load(stored_embedding_path)

    # 새로운 오디오 파일들에 대한 임베딩 생성 및 업데이트
    for audio_file in new_audio_files:
        new_audio = load_audio(audio_file)
        new_embedding = generate_embedding(model, new_audio)
        current_embedding = update_embedding(current_embedding, new_embedding, weight)

    # 업데이트된 임베딩 저장
    np.save(output_path, current_embedding)
    print(f"Updated embedding saved to {output_path}")

def main():
    model = load_driver_recognition_model()
    audio_dir = "/home/jaehun/redimnet/jaehun"
    stored_embedding_path = "./voice_embeddings/jaehun_embedding.npy"
    new_audio_files = [os.path.join(audio_dir, audio) for audio in os.listdir(audio_dir)]
    output_path = "./voice_embeddings/jaehun_embedding.npy"
    weight = 0.7
    update_voice_profile(stored_embedding_path, new_audio_files, output_path, model, weight)

if __name__ == "__main__":
    main()