import os
import torch

def load_driver_recognition_model() -> torch.model:
    if not os.path.exists('redimnet_b0.pth'):
        model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
        torch.save(model.state_dict(), './redimnet_b0.pth')
    else:
        model = load_model('./redimnet_b0.pth')
        
    return model
