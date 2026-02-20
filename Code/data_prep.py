"""
Data Preparation Module for Amplitude Embedding Ablation Study
Bachelor Thesis: Paul Kreppold
Version: 4.3 - Lückenlose Integration für Quanten-Ablation
"""

import torch
import pandas as pd
import os
import numpy as np
import cv2  # Für hochwertiges Resizing (INTER_AREA)
from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
import config
from utils import cleanup_memory

class PneumoniaDataset(Dataset):
    """Dataset-Klasse zur Speicherung der (skalierten) Bild-Tensoren und Labels."""
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx], self.labels[idx]


class FederatedAmplitudeOrchestrator:
    def __init__(self, num_qubits=10, batch_size=16, experiment_seed=42):
        self.num_qubits = num_qubits
        self.batch_size = batch_size
        self.exp_seed = experiment_seed
        self.calib_seed = config.CALIBRATION_SEED

        # Physikalische Kapazität des Hilbert-Raums (2^n)
        self.max_capacity = 2 ** num_qubits

        # Berechnung der quadratischen Kantenlänge für das Resizing
        # 2^8 = 256 -> 16x16 Pixel | 2^6 = 64 -> 8x8 Pixel
        self.side_length = int(np.sqrt(self.max_capacity))

        self.raw_pools = {}
        self.metadata = {}

    def _resize_image_data(self, data):
        """
        Dynamische Anpassung der Bildauflösung an die Qubit-Kapazität.
        Wichtig: Bei 10 Qubits (1024) passt das 28x28 (784) Bild ohne Resizing.
        """
        imgs = data.reshape(-1, 28, 28)
        resized_imgs = []

        if self.max_capacity >= 784:
            # Original beibehalten, Padding macht AmplitudeEmbedding(pad_with=0.0)
            for img in imgs:
                resized_imgs.append(img.flatten())
        else:
            # Physikalische Verkleinerung notwendig
            for img in imgs:
                res = cv2.resize(img, (self.side_length, self.side_length),
                                 interpolation=cv2.INTER_AREA)
                resized_imgs.append(res.flatten())

        return np.array(resized_imgs)

    def _get_disjoint_sample(self, df, target_n, pos_ratio, custom_seed=None):
        s = custom_seed if custom_seed is not None else self.calib_seed
        n_pos = int(target_n * pos_ratio)
        n_neg = target_n - n_pos

        pos_df = df[df['label'] == 1].sample(n=min(n_pos, len(df[df['label']==1])), random_state=s)
        neg_df = df[df['label'] == 0].sample(n=min(n_neg, len(df[df['label']==0])), random_state=s)

        sample = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=s)
        return sample, df.drop(sample.index)

    def _load_data_flexible(self, path, dataset_type):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Datensatz {dataset_type} nicht gefunden unter: {abs_path}")

        if abs_path.endswith(".npz"):
            data = np.load(abs_path)
            x = data['x'].reshape(len(data['x']), -1) / 255.0  # Normalisierung
            y = data['y'].squeeze()
            return pd.DataFrame(x).assign(label=y)
        return pd.DataFrame()

    def _prepare_mnist(self):
        DataClass = getattr(medmnist, INFO['pneumoniamnist']['python_class'])
        data = [pd.DataFrame(DataClass(split=s, download=True).imgs.reshape(-1, 784)/255.0).assign(label=DataClass(split=s).labels.squeeze()) for s in ['train', 'val', 'test']]
        df = pd.concat(data).drop_duplicates().reset_index(drop=True)

        _, rem = self._get_disjoint_sample(df, 250, 0.50, custom_seed=self.calib_seed)
        test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.75, custom_seed=self.calib_seed)

        for i in range(1, 5):
            sub_seed = (self.exp_seed + i)
            tr, rem = self._get_disjoint_sample(rem, 1000, 0.90 if i == 1 else 0.75, custom_seed=sub_seed)
            self.raw_pools[f"client_1_sub_{i}"] = {
                'train_x': tr.drop(columns='label').values, 'train_y': tr['label'].values,
                'val_x': val_df.drop(columns='label').values, 'val_y': val_df['label'].values,
                'test_x': test_df.drop(columns='label').values, 'test_y': test_df['label'].values
            }

    def _prepare_rsna(self, root):
        df = self._load_data_flexible(root, "rsna")
        c2_pool = df.sample(frac=0.5, random_state=self.calib_seed)
        c3_pool = df.drop(c2_pool.index)

        for bid_idx, (bid, pool, pr) in enumerate([('client_2', c2_pool, 0.40), ('client_3', c3_pool, 0.60)]):
            _, rem = self._get_disjoint_sample(pool, 125, 0.50, custom_seed=self.calib_seed)
            test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
            val_df, rem = self._get_disjoint_sample(rem, 524, pr, custom_seed=self.calib_seed)

            for i in range(1, 5):
                sub_seed = (self.exp_seed + 10 + (bid_idx * 4) + i)
                tr, rem = self._get_disjoint_sample(rem, 1000, pr, custom_seed=sub_seed)
                self.raw_pools[f"{bid}_sub_{i}"] = {
                    'train_x': tr.drop(columns='label').values, 'train_y': tr['label'].values,
                    'val_x': val_df.drop(columns='label').values, 'val_y': val_df['label'].values,
                    'test_x': test_df.drop(columns='label').values, 'test_y': test_df['label'].values
                }

    def _prepare_chexpert(self, root):
        df = self._load_data_flexible(root, "chexpert")
        _, rem = self._get_disjoint_sample(df, 250, 0.50, custom_seed=self.calib_seed)
        test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.25, custom_seed=self.calib_seed)

        for i in range(1, 5):
            sub_seed = (self.exp_seed + 30 + i)
            tr, rem = self._get_disjoint_sample(rem, 1000, 0.10 if i == 1 else 0.25, custom_seed=sub_seed)
            self.raw_pools[f"client_4_sub_{i}"] = {
                'train_x': tr.drop(columns='label').values, 'train_y': tr['label'].values,
                'val_x': val_df.drop(columns='label').values, 'val_y': val_df['label'].values,
                'test_x': test_df.drop(columns='label').values, 'test_y': test_df['label'].values
            }

    def build(self, rsna_root, chexpert_root, verbose=False):
        self._prepare_mnist()
        self._prepare_rsna(rsna_root)
        self._prepare_chexpert(chexpert_root)

        if verbose:
            print(f"--- Amplitude Embedding Setup ---")
            print(f"Qubits: {self.num_qubits} | Kapazität: {self.max_capacity}")
            print(f"Resizing auf: {self.side_length}x{self.side_length} Pixel")

        final_loaders = {}
        for sid, d in self.raw_pools.items():
            def make_l(x, y, shuf=False):
                # Hier wird das Bild an die Qubit-Zahl angepasst
                x_resized = self._resize_image_data(x)
                return DataLoader(
                    PneumoniaDataset(torch.tensor(x_resized, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32).view(-1, 1)),
                    batch_size=self.batch_size, shuffle=shuf, drop_last=True
                )

            final_loaders[sid] = {
                'train': make_l(d['train_x'], d['train_y'], True),
                'val': make_l(d['val_x'], d['val_y']),
                'test': make_l(d['test_x'], d['test_y'])
            }
            self.metadata[sid] = {'minority_idx': np.argmin(np.bincount(d['train_y'].astype(int)))}

        self.raw_pools = {}
        cleanup_memory()
        return final_loaders, self.metadata

# --- Hilfsfunktionen für den Main Loop ---

def get_federated_amplitude_loaders(rsna_root, chexpert_root, current_seed=42, num_qubits=10, verbose=False):
    orchestrator = FederatedAmplitudeOrchestrator(num_qubits=num_qubits, batch_size=config.BATCH_SIZE, experiment_seed=current_seed)
    return orchestrator.build(rsna_root, chexpert_root, verbose=verbose)

def get_centralized_loader(all_fed_loaders, batch_size):
    """Kombiniert Client-Trainingssets für die zentrale Baseline der Ablation."""
    xs, ys = [], []
    for s in all_fed_loaders.values():
        xs.append(s['train'].dataset.dataset)
        ys.append(s['train'].dataset.labels)
    return DataLoader(PneumoniaDataset(torch.cat(xs), torch.cat(ys)), batch_size=batch_size, shuffle=True, drop_last=True)