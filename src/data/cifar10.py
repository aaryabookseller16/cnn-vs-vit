import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

TRAIN_SIZE = 45000
VAL_SIZE = 5000
SEED = 42
DATA_DIR = "data"

def load_cifar10_datasets():
    train_tfms = transforms.ToTensor()
    tesr_tfms = transforms.ToTensor()
    
    full_train = datasets.CIFAR10(
        root = DATA_DIR,
        train = True,
        download = True,
        transform = train_tfms
    )
    test = datasets.CIFAR10(
        root = DATA_DIR,
        train = False,
        download = True,
        transform = tesr_tfms
    )
    
    return full_train, test

def split_train_val(full_train):
    assert TRAIN_SIZE + VAL_SIZE == len(full_train) , "Split sizes must sum to 50,000"
    
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [TRAIN_SIZE, VAL_SIZE], generator=g)
    return train_ds, val_ds

def get_splits():
    full_train, test_ds = load_cifar10_datasets() # get train and test from the dataset directly
    train_ds, val_ds = split_train_val(full_train) # split train and val
    return train_ds, val_ds, test_ds



