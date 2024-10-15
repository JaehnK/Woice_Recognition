import numpy as np
import torch
import torch.nn as nn
import torchaudio
from typing import Tuple
from pathlib import Path
import os
import time
import sys
import threading


def loading_animation_thread(stop_event):
    animation = "|/-\\"
    i = 0
    while not stop_event.is_set():
        sys.stdout.write("\r\033[1;33mLoading: " + animation[i % len(animation)] + "\033[0m")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r\033[1;33mLoading: done\033[0m\n")

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

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def compare_voices(stored_embedding_path: str, new_audio_path: str, threshold: float = 0.7) -> Tuple[bool, float]:
    print("Starting Speaker Identification ...")
    # 저장된 임베딩 로드
    stored_embedding = np.load(stored_embedding_path)

    # 새로운 음성 파일 로드 및 임베딩 생성
    new_audio = load_audio(new_audio_path)
    new_embedding = generate_embedding(model, new_audio)

    # 코사인 유사도 계산
    similarity = cosine_similarity(stored_embedding, new_embedding)

    # 유사도가 임계값을 넘으면 True, 아니면 False 반환
    is_same_speaker = similarity > threshold

    return is_same_speaker, similarity

def loading_animation() -> None:
    animation = "|/-\\"
    for i in range(10):
        time.sleep(0.1)
        sys.stdout.write("\r\033[1;33mLoading: " + animation[i % len(animation)] + "\033[0m")
        sys.stdout.flush()
    sys.stdout.write("\r\033[1;33mLoading: done\033[0m\n")
    time.sleep(0.5)

def compare_voices_with_animation(stored_embedding_path: str, input_audio: torch.Tensor, 
                                    model: nn.Module, threshold: float = 0.7) -> Tuple[bool, float]:
    
    # 스레드 중지 이벤트 생성
    stop_animation = threading.Event()
    
    # 로딩 애니메이션 스레드 시작
    animation_thread = threading.Thread(target=loading_animation_thread, args=(stop_animation,))
    animation_thread.start()
    
    try:
        # 실제 비교 작업 수행
        stored_embedding = np.load(stored_embedding_path)
        # new_audio = load_audio(new_audio_path)
        new_embedding = generate_embedding(model, input_audio)
        similarity = cosine_similarity(stored_embedding, new_embedding)
        is_same_speaker = similarity > threshold
        
        # 분석 완료 후 최소 1초 대기 (애니메이션이 한 바퀴 이상 돌도록)
        time.sleep(1)
    finally:
        # 작업 완료 후 애니메이션 중지
        stop_animation.set()
        animation_thread.join()
    
    return is_same_speaker, similarity

def compare_voices_stdout(stored_embedding_path: str, input_audio_wave: torch.Tensor, model: nn.Module):
    threshold: float = 0.6
    driver_names: list[str] = list(set([f.split('_')[0] for f in os.listdir(stored_embedding_path)]))
    
    for name in driver_names:
        stored_embedding_path_with_name = os.path.join(stored_embedding_path, f'{name}_embedding.npy')
        is_same, similarity = compare_voices_with_animation(stored_embedding_path_with_name, input_audio_wave, 
                                                                model, threshold)
        if is_same:
            print(f"\033[32mSimilarity: {similarity:.4f}\033[0m")
            print(f"\033[1;32mWoice System: Success({name})\033[0m")
            return 0, name
    if is_same:
        return 0, name
    else:
        print(name)
        print(f"\033[31mSimilarity: {similarity:.4f}\033[0m")
        print("\033[1;31mWoice System: Fail\033[0m")
        return (1, None)
    

# if __name__ == "__main__":
#     model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
#     stored_embedding_path = '/home/jaehun/redimnet/voice_embeddings'
#     new_audio_path = '/home/jaehun/redimnet/hb/hb.wav'
#     names = list(set([f.split('_')[0] for f in os.listdir('./voice_embeddings')]))
#     print(names)
    
    # compare_voices_stdout(stored_embedding_path, new_audio_path) 
