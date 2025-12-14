from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset
from medmnist import PneumoniaMNIST
import torch
from torch.utils.data import DataLoader

BASE_DIR = '/Users/paulkreppold/Local_datasets'
RSNA_PATH = os.path.join(BASE_DIR, 'RSNA_Pneumonia_Detection_Challenge')
CHEXPERT_PATH = os.path.join(BASE_DIR, 'CheXpert_Frontal_Processed')
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset_mendely_data')




def load_data_pca(batch_size=32, n_components=10):
    """
    Lädt PneumoniaMNIST, flacht es ab und reduziert es via PCA auf n_components.
    """
    print(f"[PCA] Starte Datenladung und Reduktion auf {n_components} Features...")

    # 1. Daten "roh" laden (als Numpy Arrays)
    # Wir nutzen hier die Bibliotheks-Funktion, um direkt an die Daten zu kommen
    train_dataset = PneumoniaMNIST(split='train', download=True, size=28)
    val_dataset = PneumoniaMNIST(split='val', download=True, size=28)
    test_dataset = PneumoniaMNIST(split='test', download=True, size=28)

    # Bilder extrahieren und flatten (N, 784)
    # .imgs ist (N, 28, 28) -> reshape zu (N, 784)
    X_train = train_dataset.imgs.reshape(len(train_dataset), -1).astype(np.float32)
    X_val = val_dataset.imgs.reshape(len(val_dataset), -1).astype(np.float32)
    X_test = test_dataset.imgs.reshape(len(test_dataset), -1).astype(np.float32)

    y_train = train_dataset.labels.astype(np.float32)  # Labels behalten
    y_val = val_dataset.labels.astype(np.float32)
    y_test = test_dataset.labels.astype(np.float32)

    # 2. PCA anwenden
    # WICHTIG: Fit nur auf TRAIN Daten, um Data Leakage zu vermeiden!
    pca = PCA(n_components=n_components)

    print("   -> Fitte PCA auf Trainingsdaten...")
    X_train_pca = pca.fit_transform(X_train)

    # Transform auf Val und Test anwenden (mit der Matrix von Train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # 3. Skalierung für AngleEmbedding
    # Wir wollen Winkel zwischen 0 und Pi (oder -Pi bis Pi).
    # AngleEmbedding mag normierte Werte.
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_train_pca = scaler.fit_transform(X_train_pca)
    X_val_pca = scaler.transform(X_val_pca)
    X_test_pca = scaler.transform(X_test_pca)

    print(f"   -> Datenform nach PCA: {X_train_pca.shape}")
    print(f"   -> Erklärte Varianz (Info-Gehalt): {np.sum(pca.explained_variance_ratio_):.2%}")

    # 4. Zurück in PyTorch Tensoren und DataLoader
    # WICHTIG: Labels für BCE müssen [N, 1] sein
    train_ds = TensorDataset(torch.tensor(X_train_pca), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val_pca), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test_pca), torch.tensor(y_test))

    loaders_train = {"client_1": DataLoader(train_ds, batch_size=batch_size, shuffle=True)}
    loaders_val = {"client_1": DataLoader(val_ds, batch_size=batch_size, shuffle=False)}
    loaders_test = {"client_1": DataLoader(test_ds, batch_size=batch_size, shuffle=False)}

    return loaders_train, loaders_val, loaders_test