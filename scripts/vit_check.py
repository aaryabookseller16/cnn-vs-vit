import torch
from src.models.vit import TinyViT

model = TinyViT(num_classes=10)
x = torch.randn(4, 3, 32, 32)
y = model(x)

print("Output shape:", y.shape)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", params)