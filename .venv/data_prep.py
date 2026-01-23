import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import config

# ==================================================
# 1. PYTORCH DATASET KLASSE
# ==================================================

class PneumoniaDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

# ==================================================
# 2. FEDERATED DATA ORCHESTRATOR
# ==================================================

class FederatedDataOrchestrator:
    def __init__(self, n_components=20, batch_size=16, seed=42):
        self.n_components = n_components
        self.batch_size = batch_size
        self.seed = seed # Dynamischer Seed für Varianz pro Experiment-Run
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.raw_pools = {}
        self.metadata = {}

    def _get_disjoint_sample(self, df, target_n, pos_ratio):
        """Zieht Stichproben basierend auf dem Instanz-Seed."""
        n_pos = int(target_n * pos_ratio)
        n_neg = target_n - n_pos

        pos_pool = df[df['label'] == 1]
        neg_pool = df[df['label'] == 0]

        # Nutzt den übergebenen Seed für echtes Resampling pro Seed-Durchlauf
        sampled_pos = pos_pool.sample(n=min(n_pos, len(pos_pool)), random_state=self.seed)
        sampled_neg = neg_pool.sample(n=min(n_neg, len(neg_pool)), random_state=self.seed)

        sample = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=self.seed)
        remaining_df = df.drop(sample.index)
        return sample, remaining_df

    def _load_images_from_disk(self, df):
        imgs, lbls = [], []
        for _, row in df.iterrows():
            if os.path.exists(row['full_path']):
                img = Image.open(row['full_path']).convert('L')
                imgs.append(np.array(self.transform(img)).flatten())
                lbls.append(row['label'])
        return np.array(imgs, dtype=np.float32), np.array(lbls, dtype=np.float32)

    # --- DATENSATZ-LOGIK MIT OPTIMIERTEM SHARING ---

    def _prepare_mnist(self):
        """Client 1: PneumoniaMNIST (Test/Val außerhalb der Schleife fixiert)."""
        info = INFO['pneumoniamnist']
        DataClass = getattr(medmnist, info['python_class'])
        data_list = []
        for s in ['train', 'val', 'test']:
            ds = DataClass(split=s, download=True)
            tmp = pd.DataFrame(ds.imgs.reshape(-1, 784) / 255.0)
            tmp['label'] = ds.labels.squeeze()
            data_list.append(tmp)

        full_df = pd.concat(data_list).drop_duplicates().reset_index(drop=True)

        # Gemeinsame Test/Val-Sets für alle MNIST-Subclients
        test_df, rem = self._get_disjoint_sample(full_df, 624, 0.50)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.75)

        tx, ty = test_df.drop(columns='label').values, test_df['label'].values
        vx, vy = val_df.drop(columns='label').values, val_df['label'].values

        for i in range(1, 5):
            ratio = 0.90 if i == 1 else 0.75
            train_df, rem = self._get_disjoint_sample(rem, 1000, ratio)
            sid = f"client_1_sub_{i}"
            self.raw_pools[sid] = {
                'train_x': train_df.drop(columns='label').values, 'train_y': train_df['label'].values,
                'val_x': vx, 'val_y': vy, 'test_x': tx, 'test_y': ty
            }

    def _prepare_rsna(self, root):
        """Client 2 & 3: RSNA (Behebt Punkt 1 & 2 aus dem Anhang)."""
        df = pd.read_csv(os.path.join(root, 'stage2_train_metadata.csv'))
        df = df[df['class'].isin(['Normal', 'Lung Opacity'])].drop_duplicates(subset='patientId')
        df['label'] = df['class'].map({'Normal': 0, 'Lung Opacity': 1})
        df['full_path'] = df['patientId'].apply(lambda x: os.path.join(root, 'Training/Images', f"{x}.png"))

        c2_pool = df.sample(frac=0.5, random_state=self.seed)
        c3_pool = df.drop(c2_pool.index)

        for base_id, pool, ratio in [('client_2', c2_pool, 0.60), ('client_3', c3_pool, 0.40)]:
            # 1. Gemeinsame Test/Val Sets für die Subclients dieses Base-Clients ziehen
            test_df, rem = self._get_disjoint_sample(pool, 624, 0.50)
            val_df, rem = self._get_disjoint_sample(rem, 524, ratio)

            # 2. Bilder EINMAL laden (behebt 🐌-Problem)
            vx, vy = self._load_images_from_disk(val_df)
            tx, ty = self._load_images_from_disk(test_df)

            for i in range(1, 5):
                train_df, rem = self._get_disjoint_sample(rem, 1000, ratio)
                x, y = self._load_images_from_disk(train_df)
                self.raw_pools[f"{base_id}_sub_{i}"] = {
                    'train_x': x, 'train_y': y, 'val_x': vx, 'val_y': vy, 'test_x': tx, 'test_y': ty
                }

    def _prepare_chexpert(self, root):
        """Client 4: CheXpert (Optimiertes Sharing)."""
        df = pd.read_csv(os.path.join(root, 'train.csv'))
        df = df[(df['Pneumonia'] == 1.0) | (df['No Finding'] == 1.0)]
        df['label'] = (df['Pneumonia'] == 1.0).astype(int)
        df['full_path'] = df['Path'].apply(lambda x: os.path.join(root, x))

        test_df, rem = self._get_disjoint_sample(df, 624, 0.50)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.25)

        vx, vy = self._load_images_from_disk(val_df)
        tx, ty = self._load_images_from_disk(test_df)

        for i in range(1, 5):
            ratio = 0.10 if i == 1 else 0.25
            train_df, rem = self._get_disjoint_sample(rem, 1000, ratio)
            x, y = self._load_images_from_disk(train_df)
            self.raw_pools[f"client_4_sub_{i}"] = {
                'train_x': x, 'train_y': y, 'val_x': vx, 'val_y': vy, 'test_x': tx, 'test_y': ty
            }

    def build(self, rsna_root, chexpert_root):
        """Führt PCA & Scaling basierend auf dem aktuellen Seed-Pool aus."""
        self._prepare_mnist()
        self._prepare_rsna(rsna_root)
        self._prepare_chexpert(chexpert_root)

        all_train_x = np.vstack([v['train_x'] for v in self.raw_pools.values()])
        pca = PCA(n_components=self.n_components).fit(all_train_x)
        scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(all_train_x))

        final_loaders = {}
        for sid, d in self.raw_pools.items():
            def to_loader(x, y, shuffle=False):
                x_trans = scaler.transform(pca.transform(x))
                ds = PneumoniaDataset(
                    torch.tensor(x_trans, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32).view(-1, 1)
                )
                return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

            final_loaders[sid] = {
                'train': to_loader(d['train_x'], d['train_y'], True),
                'val': to_loader(d['val_x'], d['val_y']),
                'test': to_loader(d['test_x'], d['test_y'])
            }
            y_train = d['train_y'].astype(int)
            counts = np.bincount(y_train, minlength=2)
            self.metadata[sid] = {'lds_ratio': np.max(counts) / len(y_train), 'minority_idx': np.argmin(counts)}

        return final_loaders, self.metadata

# ==================================================
# 3. EXTERNE SCHNITTSTELLEN
# ==================================================

def get_federated_pca_loaders(rsna_root, chexpert_root, current_seed=42):
    """Schnittstelle für die main.py. Erzeugt pro Seed einen neuen Split."""
    orchestrator = FederatedDataOrchestrator(
        n_components=config.NUM_FEATURES,
        batch_size=config.BATCH_SIZE,
        seed=current_seed
    )
    return orchestrator.build(rsna_root, chexpert_root)

def get_centralized_loader(all_fed_loaders, batch_size):
    """Bündelt alle Trainingsdaten der Subclients für die zentrale Baseline."""
    all_x = []
    all_y = []
    for sid in all_fed_loaders:
        # Zugriff auf die Tensors im PneumoniaDataset
        ds = all_fed_loaders[sid]['train'].dataset
        all_x.append(ds.dataset)
        all_y.append(ds.labels)

    combined_ds = PneumoniaDataset(torch.cat(all_x, dim=0), torch.cat(all_y, dim=0))
    return DataLoader(combined_ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ==================================================
# 4. ANALYSE- UND VALIDIERUNGSFUNKTIONEN
# ==================================================

def print_dataset_stats(all_fed_loaders):
    """Gibt die Klassenverteilung pro Subclient aus."""
    print("\n" + "=" * 95)
    print(f"{'SUBCLIENT DATEN-VERTEILUNG':^95}")
    print("=" * 95)
    print(f"{'Subclient ID':<20} | {'Split':<6} | {'Gesamt':<7} | {'Normal':<7} | {'Pneu':<7} | {'Anteil Pneu'}")
    print("-" * 95)
    for sid, splits in all_fed_loaders.items():
        for split_name in ['train', 'val', 'test']:
            ds = splits[split_name].dataset
            labels = ds.labels.flatten()
            total = len(labels)
            pos = torch.sum(labels == 1).item()
            neg = torch.sum(labels == 0).item()
            ratio = (pos / total * 100) if total > 0 else 0
            display_id = sid if split_name == 'train' else ""
            print(f"{display_id:<20} | {split_name:<6} | {total:<7} | {int(neg):<7} | {int(pos):<7} | {ratio:>10.1f}%")
        print("-" * 95)


def verify_disjoint_subclients(final_loaders):
    """Validiert mathematisch, dass keine Bilder doppelt vorkommen."""
    seen_hashes = {}
    overlap = False
    for sid, splits in final_loaders.items():
        x_data = splits['train'].dataset.dataset.numpy()
        for i in range(len(x_data)):
            h = hash(x_data[i].tobytes())
            if h in seen_hashes:
                print(f"ALARM: Überlappung in {sid} gefunden! (Zuvor in {seen_hashes[h]})")
                overlap = True
            else:
                seen_hashes[h] = sid
    if not overlap:
        print(f"ERGEBNIS: Alle {len(final_loaders)} Subclients sind zu 100% disjunkt.")
