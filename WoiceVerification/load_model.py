import os
import torch
import torch.nn as nn

# def load_driver_recognition_model() -> nn.Module:
#     if not os.path.exists('redimnet_b0.pth'):
#         print("Download DRM model...")
#         model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True, verbose = False)
#         torch.save(model.state_dict(), './redimnet_b0.pth')
#     else:
#         model = torch.load('./redimnet_b0.pth')
    
#     print("Load Model Success")
#     return model

def load_driver_recognition_model() -> nn.Module:
    if not os.path.exists('redimnet_b0.pth'):
        print("Download DRM model...")
        model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True, verbose=False)
        torch.save(model.state_dict(), './redimnet_b0.pth')
    else:
        print("Loading model from local file...")
        model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=False, finetuned=False, verbose=False)
        model.load_state_dict(torch.load('./redimnet_b0.pth'))
    
    print("Load Model Success")
    return model