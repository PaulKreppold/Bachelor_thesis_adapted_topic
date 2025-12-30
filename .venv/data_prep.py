import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_pneumonia_mnist_loaders(batch_size=32, val_split=0.1):
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Transform-Pipeline (Resize auf 8x8 = 64 Features -> 6 Qubits)
    transform = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]) # Empfohlen für MedMNIST
    ])

    # Datensätze laden
    # MedMNIST hat bereits vordefinierte Splits (train, val, test)
    # Wir laden hier train und test und bauen den val-split wie gewünscht manuell
    train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    # Da PneumoniaMNIST bereits binär ist, brauchen wir kein get_binary_indices!
    # Wir nutzen den vordefinierten Trainingssatz und splitten ihn in Train/Val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader