import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
from PIL import Image
from sklearn.model_selection import train_test_split
from config import clients, BATCH_SIZE

# -----------------------------
# Konstanten
# -----------------------------
# Wir nutzen die Gesamtgröße von PneumoniaMNIST als Referenz für RSNA Downsampling
TARGET_TOTAL_SAMPLES = 5856


def get_rsna_metadata_split(metadata_csv):
    """Teilt die RSNA Metadaten strikt nach Patienten-IDs auf zwei disjunkte Pools auf."""
    df = pd.read_csv(metadata_csv)
    # Sicherstellen, dass Duplikate (mehrere Boxen pro Patient) die ID-Trennung nicht korrumpieren
    patient_ids = df['patientId'].unique()
    ids_c2, ids_c3 = train_test_split(patient_ids, test_size=0.5, random_state=42)

    # Filtern der Dataframes basierend auf den IDs
    df_c2 = df[df['patientId'].isin(ids_c2)].drop_duplicates(subset=['patientId']).copy()
    df_c3 = df[df['patientId'].isin(ids_c3)].drop_duplicates(subset=['patientId']).copy()
    return df_c2, df_c3


# -----------------------------
# Datasets
# -----------------------------
class RSNADataset(Dataset):
    def __init__(self, df_pool, images_dir, transform=None, ratio_normal=0.5, target_size=TARGET_TOTAL_SAMPLES):
        self.transform = transform

        # Klassen-Trennung für gezieltes Sampling
        normal_df = df_pool[df_pool['Target'] == 0]
        pneum_df = df_pool[df_pool['Target'] == 1]

        n_normal = int(target_size * ratio_normal)
        n_pneum = target_size - n_normal

        # Sampling durchführen (mit Fallback, falls Pool zu klein)
        s_normal = normal_df.sample(n=min(n_normal, len(normal_df)), random_state=42)
        s_pneum = pneum_df.sample(n=min(n_pneum, len(pneum_df)), random_state=42)

        self.df = pd.concat([s_normal, s_pneum]).reset_index(drop=True)
        self.labels = self.df['Target'].values
        self.paths = [os.path.join(images_dir, f"{pid}.png") for pid in self.df['patientId']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # RSNA Bilder sind oft sehr groß, 'L' konvertiert zu Grayscale
            img = Image.open(self.paths[idx]).convert('L')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            # Rückfalloption bei fehlenden/korrupten Dateien
            return torch.zeros((1, 32, 32)), self.labels[idx]


# -----------------------------
# Loader & Verteilung
# -----------------------------
def get_pneumonia_dataloaders(batch_size=BATCH_SIZE, num_workers=0, download=True):
    """Standard MedMNIST Pneumonia Loader für Client 1."""
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = PneumoniaMNIST(split='train', transform=train_transform, download=download)
    val_dataset = PneumoniaMNIST(split='val', transform=test_transform, download=download)
    test_dataset = PneumoniaMNIST(split='test', transform=test_transform, download=download)

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


def get_rsna_loaders(client_id, df_pool, batch_size=BATCH_SIZE, val_split=0.1, test_split=0.1):
    """Spezialisierte Loader für Client 2 und 3 mit getrennten Transforms."""
    base_dir = "/Users/paulkreppold/Local_datasets/RSNA_Pneumonia_Detection_Challenge"
    images_dir = os.path.join(base_dir, "Training/Images")

    # 1. TRAINING-Transform (mit Augmentation)
    rsna_train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(100),
        transforms.Resize((32, 32)),
        transforms.ColorJitter(contrast=0.2, brightness=0.2),  # Nur hier!
        transforms.RandomHorizontalFlip(),  # Optional: erhöht Robustheit
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. EVALUATION-Transform (Rein und konsistent)
    rsna_test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(100),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Unterschiedliche Label-Verteilung (Non-IID Simulation)
    ratio_normal = 0.6 if client_id == 'client_2' else 0.4

    # Dataset initial erstellen (noch ohne transform, um flexibel zu bleiben)
    # Oder: Wir erstellen zwei Instanzen mit derselben DF-Basis
    full_dataset = RSNADataset(df_pool, images_dir, transform=None, ratio_normal=ratio_normal)

    # Indizes splitten
    indices = np.arange(len(full_dataset))
    trainval_idx, test_idx = train_test_split(indices, test_size=test_split, stratify=full_dataset.labels,
                                              random_state=42)

    val_size_adj = val_split / (1 - test_split)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size_adj,
                                          stratify=full_dataset.labels[trainval_idx], random_state=42)

    # Subsets erstellen
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    # TRICK: Wir weisen den Subsets die spezifischen Transforms zu
    # Da Subset nur auf das Original-Dataset zeigt, überschreiben wir die Dataset-Instanz
    # für die Loader oder nutzen eine kleine Wrapper-Klasse (Sauberste Lösung)

    class ApplyTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            # Hier wenden wir die Transformation auf das PIL Image an
            # (Das Original-Dataset muss dafür das Image OHNE transform zurückgeben)
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    # Da dein RSNADataset das Transform im __getitem__ anwendet,
    # setzen wir dort transform=None und nutzen den Wrapper:
    full_dataset.transform = None

    return (DataLoader(ApplyTransform(train_subset, rsna_train_transform), batch_size=batch_size, shuffle=True),
            DataLoader(ApplyTransform(val_subset, rsna_test_transform), batch_size=batch_size, shuffle=False),
            DataLoader(ApplyTransform(test_subset, rsna_test_transform), batch_size=batch_size, shuffle=False))


def get_all_client_loaders(batch_size=BATCH_SIZE):
    """Zentrale Funktion zum Abrufen aller Client-Loader."""
    client_loaders = {}
    base_dir = "/Users/paulkreppold/Local_datasets/RSNA_Pneumonia_Detection_Challenge"
    metadata_csv = os.path.join(base_dir, "stage2_train_metadata.csv")

    # Einmaliges Splitten der RSNA Datenquelle für Client 2 und 3
    df_pool_c2, df_pool_c3 = get_rsna_metadata_split(metadata_csv)

    for client_id in clients:
        if client_id == 'client_1':
            tl, vl, tsl = get_pneumonia_dataloaders(batch_size)
        elif client_id == 'client_2':
            tl, vl, tsl = get_rsna_loaders(client_id, df_pool_c2, batch_size)
        elif client_id == 'client_3':
            tl, vl, tsl = get_rsna_loaders(client_id, df_pool_c3, batch_size)
        client_loaders[client_id] = {"train": tl, "val": vl, "test": tsl}

    return client_loaders


# -----------------------------
# Analyse-Tools
# -----------------------------
def print_class_distributions(client_loaders):
    """Gibt eine Übersicht über die Verteilung der Klassen pro Client aus."""
    data = []
    for cid, loaders in client_loaders.items():
        for split in ['train', 'val', 'test']:
            all_y = []
            # Wir iterieren durch den DataLoader, um die Labels zu zählen
            for _, labels in loaders[split]:
                all_y.extend(labels.tolist())

            all_y = np.array(all_y)
            tot = len(all_y)
            n1 = int(np.sum(all_y))
            n0 = tot - n1
            data.append([cid, split.upper(), tot, f"{n0} ({n0 / tot:.1%})", f"{n1} ({n1 / tot:.1%})"])

    df = pd.DataFrame(data, columns=['Client', 'Split', 'Total', 'Normal (0)', 'Pneumonie (1)'])
    print(f"\n{'=' * 75}\n{'ÜBERSICHT KLASSENVERTEILUNG':^75}\n{'=' * 75}")
    print(df.to_string(index=False, justify='center', col_space=12))


if __name__ == "__main__":
    # Test-Lauf
    loaders = get_all_client_loaders()
    print_class_distributions(loaders)