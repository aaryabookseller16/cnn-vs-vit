import torch
from src.models.cnn import SmallCIFARCNN

model = SmallCIFARCNN(num_classes=10, dropout=0.0)
x = torch.randn(4, 3, 32, 32)
y = model(x)

print("Output shape:", y.shape)