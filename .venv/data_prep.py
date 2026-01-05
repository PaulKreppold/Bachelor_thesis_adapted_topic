import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_pca_data_loaders(batch_size=32, val_split=0.1, n_components=12):
    """
    Lädt PneumoniaMNIST, reduziert die Dimension via PCA und skaliert die Daten
    für ein Single-Upload VQC Baseline-Modell.

    Args:
        batch_size (int): Batch-Größe für die Loader.
        val_split (float): Anteil der Validierungsdaten.
        n_components (int): Anzahl der PCA-Komponenten. Bei 6 Qubits und Dense
                           Encoding sind dies standardmäßig 12.
    """
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Transformation für MedMNIST: Konvertierung in Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datensätze laden
    full_train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)

    def extract_raw_data(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        images, targets = next(iter(loader))
        # Flatten: (N, 1, 28, 28) -> (N, 784) für PCA-Verarbeitung
        return images.view(len(dataset), -1).numpy(), targets.numpy().squeeze()

    # Daten extrahieren und für PCA vorbereiten
    x_train_raw, y_train_raw = extract_raw_data(full_train_dataset)
    x_test_raw, y_test_raw = extract_raw_data(test_dataset)

    # ==========================================
    # PCA: Informations-Effizienz
    # ==========================================
    # Reduktion auf die Kapazität des Quantenschaltkreises
    pca = PCA(n_components=n_components)

    # MinMaxScaler auf [0, pi] skaliert die Daten für Quanten-Rotationswinkel (Angle Encoding)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    # Fit nur auf Trainingsdaten zur Vermeidung von Data Leakage
    x_train_pca = pca.fit_transform(x_train_raw)
    x_train_scaled = scaler.fit_transform(x_train_pca)

    # Testdaten mit den Parametern der Trainingsdaten transformieren
    x_test_pca = pca.transform(x_test_raw)
    x_test_scaled = scaler.transform(x_test_pca)

    # ==========================================
    # PyTorch Datasets & Loader
    # ==========================================
    # Trainingstensor erstellen (Labels als Float für BCELoss Kompatibilität)
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

    # Train/Val Split zur Überwachung der Generalisierung
    val_size = int(len(train_tensor) * val_split)
    train_size = len(train_tensor) - val_size
    train_subset, val_subset = random_split(
        train_tensor, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"PCA Baseline Setup: {n_components} Komponenten extrahiert.")
    print(f"Datensatz-Größen -> Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader