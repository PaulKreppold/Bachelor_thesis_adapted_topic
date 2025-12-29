import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def get_mnist_binary_loaders(batch_size=32, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def get_binary_indices(dataset):
        return [i for i, label in enumerate(dataset.targets) if label in [0, 1]]

    # Subsets für 0 und 1 erstellen
    full_train_subset = Subset(train_dataset, get_binary_indices(train_dataset))
    test_subset = Subset(test_dataset, get_binary_indices(test_dataset))

    # Training in Train und Validation aufteilen
    val_size = int(len(full_train_subset) * val_split)
    train_size = len(full_train_subset) - val_size
    train_subset, val_subset = random_split(full_train_subset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader