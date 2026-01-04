import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_pca_data_loaders(batch_size=32, val_split=0.1, n_components=12):  # Standard auf 12 gesetzt
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    full_train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    def extract_raw_data(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        images, targets = next(iter(loader))
        # .squeeze() entfernt die überflüssige Dimension bei den Labels (N, 1) -> (N,)
        return images.view(len(dataset), -1).numpy(), targets.numpy().squeeze()

    x_train_raw, y_train_raw = extract_raw_data(full_train_dataset)
    x_test_raw, y_test_raw = extract_raw_data(test_dataset)

    # PCA & Scaling
    pca = PCA(n_components=n_components)
    # 0 bis pi ist ideal für RY/RZ-Encoding, um den Hilbert-Raum gut zu nutzen
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    x_train_pca = pca.fit_transform(x_train_raw)
    x_train_scaled = scaler.fit_transform(x_train_pca)

    x_test_pca = pca.transform(x_test_raw)
    x_test_scaled = scaler.transform(x_test_pca)

    # Tensoren erstellen (Labels als LongTensor für Klassifikation)
    train_tensor = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train_raw)
    )

    # Test Loader direkt hier erstellen
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(x_test_scaled), torch.LongTensor(y_test_raw)),
        batch_size=batch_size, shuffle=False
    )

    # Train/Val Split
    val_size = int(len(train_tensor) * val_split)
    train_size = len(train_tensor) - val_size
    train_subset, val_subset = random_split(
        train_tensor, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader