import torch
ckpt = torch.load("models/best_model.pth", map_location="cpu")
print(ckpt["classes"])