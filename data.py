"""Data loading and preprocessing shared by all scripts."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

# Light augmentation helps the model cope with real handwriting.
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    normalize,
])
eval_tf = transforms.Compose([transforms.ToTensor(), normalize])


def get_loaders(batch_size=128, val_size=5000, data_dir="data", seed=42):
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    full_val = datasets.MNIST(data_dir, train=True, download=True, transform=eval_tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    n = len(full_train)
    train_idx, val_idx = random_split(range(n), [n - val_size, val_size], generator=g)
    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    val_set = torch.utils.data.Subset(full_val, val_idx.indices)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=512, num_workers=2),
        DataLoader(test, batch_size=512, num_workers=2),
    )
