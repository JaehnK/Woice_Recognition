import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

def preprocess_audio(file_path: str) -> torch.Tensor:
    print("processing audios...")
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform

def generate_embedding(model: torch.nn.Module, audio: torch.Tensor) -> np.ndarray:
    print("Generate embeddings...")
    model.eval()
    with torch.no_grad():
        embedding = model(audio)
    return embedding.numpy().flatten()

def save_embedding(embedding: np.ndarray, file_path: str) -> None:
    np.save(file_path, embedding)

def load_embedding(file_path: str) -> np.ndarray:
    return np.load(file_path)

def save_metadata(metadata: Dict[str, Any], file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(metadata, f)

def vectorize_and_save_voice(audio_file: str, output_dir: str, speaker_name: str, secret_code: str) -> None:
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = preprocess_audio(audio_file)
    embedding = generate_embedding(model, audio)
    embedding_file = output_dir / f"{speaker_name}_embedding.npy"
    save_embedding(embedding, str(embedding_file))

    # 메타데이터 저장
    metadata: Dict[str, Any] = {
        "speaker_name": speaker_name,
        "secret_code" : secret_code,
        "original_audio": str(audio_file),
        "embedding_file": str(embedding_file),
        "model_used": "ReDimNet_b0",
        "embedding_dim": embedding.shape[0]
    }
    metadata_file = output_dir / f"{speaker_name}_metadata.json"
    save_metadata(metadata, str(metadata_file))

    print(f"Voice Vectorization Finished: {speaker_name}")
    print(f"Embedding File: {embedding_file}")
    print(f"Metadata File: {metadata_file}")

def main() -> int:
    model: torch.nn.Module = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)

    audio_file: str = "/home/jaehun/redimnet/hb/hb.wav"
    output_dir: str = "voice_embeddings"
    speaker_name: str = "hyebin"
    secret_code: str = "forty two"

    vectorize_and_save_voice(audio_file, output_dir, speaker_name, secret_code)
    return (0)

if __name__ == '__main__':
    main()
