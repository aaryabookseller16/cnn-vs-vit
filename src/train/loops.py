from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class EpochStats:
    loss: float
    acc: float
    
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1) #what class does the model think is the most likely to be correct
    return (pred == y).float().mean().item()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> EpochStats:
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device, non_blocking = True) # Send this batch to wherever the model lives, efficiently
        y = y.to(device, non_blocking = True)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs
        
    return EpochStats(loss=total_loss/ total, acc= total_correct/ total)
    
    
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochStats:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return EpochStats(loss=total_loss / total, acc=total_correct / total)
        
