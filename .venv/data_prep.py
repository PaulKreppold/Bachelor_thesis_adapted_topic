import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_pca_data_loaders(batch_size=32, val_split=0.1, n_components=48):
    """
    Lädt PneumoniaMNIST, reduziert die Dimension via PCA auf 48 Komponenten
    und skaliert diese für das Data Re-Uploading VQC Modell.

    Args:
        batch_size (int): Batch-Größe für die Loader.
        val_split (float): Anteil der Validierungsdaten.
        n_components (int): Anzahl der PCA-Komponenten (Standard 48 für 4 Re-Uploading Blöcke).
    """
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Standard Transformation für MedMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Hinweis: Normalisierung erfolgt hier noch nicht, da wir PCA auf Rohwerten machen
    ])

    # Datensätze laden
    full_train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    def extract_raw_data(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        images, targets = next(iter(loader))
        # Flatten: (N, 1, 28, 28) -> (N, 784)
        return images.view(len(dataset), -1).numpy(), targets.numpy().squeeze()

    # Daten extrahieren und flachklopfen
    x_train_raw, y_train_raw = extract_raw_data(full_train_dataset)
    x_test_raw, y_test_raw = extract_raw_data(test_dataset)

    # ==========================================
    # PCA & Scaling
    # ==========================================
    # Reduktion auf 48 Komponenten, um mehr Bildinformationen zu erhalten
    pca = PCA(n_components=n_components)

    # MinMaxScaler auf [0, pi] skaliert die Daten passend für Quanten-Rotationswinkel
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    # Fit nur auf Trainingsdaten, um Data Leakage zu vermeiden
    x_train_pca = pca.fit_transform(x_train_raw)
    x_train_scaled = scaler.fit_transform(x_train_pca)

    # Testdaten mit den Parametern der Trainingsdaten transformieren
    x_test_pca = pca.transform(x_test_raw)
    x_test_scaled = scaler.transform(x_test_pca)

    # ==========================================
    # PyTorch Datasets & Loader
    # ==========================================
    # Trainingstensor erstellen
    # Wir konvertieren Labels zu Float für BCEWithLogitsLoss Kompatibilität im Training-Loop
    train_tensor = TensorDataset(
        torch.FloatTensor(x_train_scaled),
        torch.FloatTensor(y_train_raw).view(-1, 1)
    )

    # Test Loader
    test_tensor = TensorDataset(
        torch.FloatTensor(x_test_scaled),
        torch.FloatTensor(y_test_raw).view(-1, 1)
    )
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    # Train/Val Split
    val_size = int(len(train_tensor) * val_split)
    train_size = len(train_tensor) - val_size
    train_subset, val_subset = random_split(
        train_tensor, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"PCA Setup abgeschlossen: {n_components} Komponenten.")
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader