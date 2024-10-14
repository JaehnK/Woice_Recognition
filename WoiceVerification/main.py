import torch
import compare_voice
import embedding_voice
import update_voice
import load_model

def main() -> int:
    model: torch.model = load_model() 
    input_voice: str = input("Input Voice : 1 - 10")
    
    return (0)