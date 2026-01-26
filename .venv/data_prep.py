import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle # Neu für sauberes Sampling

class PneumoniaDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

def get_centralized_pneumonia_loaders(n_components=12, batch_size=16, seed=42):
    """
    Zentralisierter Loader, begrenzt auf 1000 Trainingsbilder,
    um QFL-Datenknappheit zu simulieren.
    """
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])

    # 1. Rohdaten laden
    train_ds = DataClass(split='train', download=True)
    val_ds = DataClass(split='val', download=True)
    test_ds = DataClass(split='test', download=True)

    def preprocess(ds):
        x = ds.imgs.reshape(-1, 784).astype(np.float32) / 255.0
        y = ds.labels.astype(np.float32)
        return x, y

    x_train_all, y_train_all = preprocess(train_ds)
    x_val, y_val = preprocess(val_ds)
    x_test, y_test = preprocess(test_ds)

    # 2. Begrenzung auf 1000 Trainingsbilder
    # Wir shuffeln mit dem Seed, um repräsentative 1000 Bilder zu erhalten
    x_train_shuffled, y_train_shuffled = shuffle(
        x_train_all, y_train_all, random_state=seed
    )
    x_train = x_train_shuffled[:1000]
    y_train = y_train_shuffled[:1000]

    print(f"📊 Data Prep: Training auf {len(x_train)} Bildern begrenzt (Seed {seed})")

    # 3. Wissenschaftliche Vorverarbeitungskette
    # Wichtig: Wir fitten alles auf den 1000 Bildern (lokales Szenario)
    std_scaler = StandardScaler().fit(x_train)
    x_train_std = std_scaler.transform(x_train)
    x_val_std = std_scaler.transform(x_val)
    x_test_std = std_scaler.transform(x_test)

    pca = PCA(n_components=n_components, random_state=seed).fit(x_train_std)
    x_train_pca = pca.transform(x_train_std)
    x_val_pca = pca.transform(x_val_std)
    x_test_pca = pca.transform(x_test_std)

    mm_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(x_train_pca)

    def to_loader(x, y, shuffle_data=False):
        x_final = mm_scaler.transform(x)
        ds = PneumoniaDataset(
            torch.tensor(x_final, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_data, drop_last=True)

    return (to_loader(x_train_pca, y_train, shuffle_data=True),
            to_loader(x_val_pca, y_val),
            to_loader(x_test_pca, y_test))