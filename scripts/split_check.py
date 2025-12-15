from src.data.cifar10 import get_splits

train_ds, val_ds, test_ds = get_splits()

print("Train:", len(train_ds))
print("Val:  ", len(val_ds))
print("Test: ", len(test_ds))