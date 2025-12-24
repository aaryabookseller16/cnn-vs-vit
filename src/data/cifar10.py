import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, DataLoader

TRAIN_SIZE = 45000
VAL_SIZE = 5000
SEED = 42
DATA_DIR = "data"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def build_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    
    return train_tfms, eval_tfms

def load_cifar10_datasets(train_tfms, eval_tfms):
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
        transform = eval_tfms
    )
    
    return full_train, test

def split_train_val(full_train):
    assert TRAIN_SIZE + VAL_SIZE == len(full_train) , "Split sizes must sum to 50,000"
    
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [TRAIN_SIZE, VAL_SIZE], generator=g)
    return train_ds, val_ds

def get_splits():
    train_tfms, eval_tfms = build_transforms()
    full_train, test_ds = load_cifar10_datasets(train_tfms=train_tfms, eval_tfms=eval_tfms) # get train and test from the dataset directly
    train_ds, val_ds = split_train_val(full_train) # split train and val
    return train_ds, val_ds, test_ds

def get_splits_fixed_transforms():
    train_tfms, eval_tfms = build_transforms()
    
    train_base = datasets.CIFAR10(
        root=DATA_DIR, train = True, download=True,transform=train_tfms
    )
    eval_base = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=eval_tfms
    )
    test_ds= datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=eval_tfms
    )
    
    assert TRAIN_SIZE + VAL_SIZE == len(train_base)
    g = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(range(len(train_base)), [TRAIN_SIZE, VAL_SIZE], generator=g)
    
    train_ds = Subset(train_base, train_subset.indices)
    val_ds = Subset(eval_base, val_subset.indices)
    
    return train_ds, val_ds, test_ds

def make_loaders(batch_size =128, num_workers=2):
    train_ds, val_ds, test_ds = get_splits_fixed_transforms()
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) 
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory = torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader



