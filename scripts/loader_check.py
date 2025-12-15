from src.data.cifar10 import make_loaders

train_loader, val_loader, test_loader = make_loaders(batch_size=128)

xb, yb = next(iter(train_loader))
print("Train batch:", xb.shape, yb.shape)

xb, yb = next(iter(val_loader))
print("Val batch:  ", xb.shape, yb.shape)

xb, yb = next(iter(test_loader))
print("Test batch: ", xb.shape, yb.shape)