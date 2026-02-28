"""
Data Preparation Module for Quantum Federated Learning
Bachelor Thesis: Paul Kreppold
Version: 3.0 - Fixed PCA, Decoupled Subclients & Consistency
"""

import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import config

class PneumoniaDataset(Dataset):
    """Standard Dataset für Tensorspeicherung nach PCA-Transformation."""
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx], self.labels[idx]

class FederatedDataOrchestrator:
    def __init__(self, n_components=20, batch_size=16, experiment_seed=42):
        self.n_components = n_components
        self.batch_size = batch_size
        self.exp_seed = experiment_seed
        # Fixiert das Koordinatensystem (PCA) & Benchmark-Basis (Test/Val)
        self.calib_seed = config.CALIBRATION_SEED

        self.raw_pools = {}
        self.metadata = {}
        self.pca_pool = []
        # Statistiken für Dokumentation
        self.pca_sources = {"MNIST": 0, "RSNA": 0, "CheXpert": 0}
        self.pca_class_counts = {0: 0, 1: 0}

    def _get_disjoint_sample(self, df, target_n, pos_ratio, custom_seed=None):
        """
        Zieht Samples und gibt den Rest zurück (Pool-Shrinking für Disjunktheit).
        custom_seed ermöglicht statistische Unabhängigkeit zwischen Clients.
        """
        s = custom_seed if custom_seed is not None else self.calib_seed

        n_pos = int(target_n * pos_ratio)
        n_neg = target_n - n_pos

        # Sampling (Zufälligkeit kontrolliert durch s)
        pos_df = df[df['label'] == 1].sample(n=min(n_pos, len(df[df['label']==1])), random_state=s)
        neg_df = df[df['label'] == 0].sample(n=min(n_neg, len(df[df['label']==0])), random_state=s)

        sample = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=s)
        return sample, df.drop(sample.index)

    def _update_pca_stats(self, df, source_name):
        self.pca_pool.append(df.drop(columns='label').values)
        self.pca_sources[source_name] += len(df)
        self.pca_class_counts[0] += len(df[df['label'] == 0])
        self.pca_class_counts[1] += len(df[df['label'] == 1])

    def _load_data_flexible(self, path, dataset_type):
        if os.path.isfile(path) and path.endswith(".npz"):
            data = np.load(path)
            x = data['x'].reshape(len(data['x']), -1)
            return pd.DataFrame(x).assign(label=data['y'].squeeze())
        return pd.DataFrame()

    def _prepare_mnist(self):
        DataClass = getattr(medmnist, INFO['pneumoniamnist']['python_class'])
        data = [pd.DataFrame(DataClass(split=s, download=True).imgs.reshape(-1, 784)/255.0).assign(label=DataClass(split=s).labels.squeeze()) for s in ['train', 'val', 'test']]
        df = pd.concat(data).drop_duplicates().reset_index(drop=True)

        # PCA & Benchmark-Sets (Test/Val) nutzen stabilen CALIBRATION_SEED
        pca_df, rem = self._get_disjoint_sample(df, 250, 0.50, custom_seed=self.calib_seed)
        self._update_pca_stats(pca_df, "MNIST")
        test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.75, custom_seed=self.calib_seed)

        for i in range(1, 5):
            # DECOUPLING: Subclient-Seed wird deriviert, um künstliche Korrelation zu vermeiden
            sub_seed = (self.exp_seed + i)
            pr = 0.90 if i == 1 else 0.75

            # rem wird sequentiell verringert -> physische Disjunktheit
            tr, rem = self._get_disjoint_sample(rem, 1000, pr, custom_seed=sub_seed)
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
            pca_df, rem = self._get_disjoint_sample(pool, 125, 0.50, custom_seed=self.calib_seed)
            self._update_pca_stats(pca_df, "RSNA")
            test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
            val_df, rem = self._get_disjoint_sample(rem, 524, pr, custom_seed=self.calib_seed)

            for i in range(1, 5):
                # Unabhängiger Seed pro Subclient (mit Offset zur Vermeidung von Kollisionen)
                sub_seed = (self.exp_seed + 10 + (bid_idx * 4) + i)
                tr, rem = self._get_disjoint_sample(rem, 1000, pr, custom_seed=sub_seed)
                self.raw_pools[f"{bid}_sub_{i}"] = {
                    'train_x': tr.drop(columns='label').values, 'train_y': tr['label'].values,
                    'val_x': val_df.drop(columns='label').values, 'val_y': val_df['label'].values,
                    'test_x': test_df.drop(columns='label').values, 'test_y': test_df['label'].values
                }

    def _prepare_chexpert(self, root):
        df = self._load_data_flexible(root, "chexpert")
        pca_df, rem = self._get_disjoint_sample(df, 250, 0.50, custom_seed=self.calib_seed)
        self._update_pca_stats(pca_df, "CheXpert")
        test_df, rem = self._get_disjoint_sample(rem, 624, 0.50, custom_seed=self.calib_seed)
        val_df, rem = self._get_disjoint_sample(rem, 524, 0.25, custom_seed=self.calib_seed)

        for i in range(1, 5):
            sub_seed = (self.exp_seed + 30 + i)
            pr = 0.10 if i == 1 else 0.25
            tr, rem = self._get_disjoint_sample(rem, 1000, pr, custom_seed=sub_seed)
            self.raw_pools[f"client_4_sub_{i}"] = {
                'train_x': tr.drop(columns='label').values, 'train_y': tr['label'].values,
                'val_x': val_df.drop(columns='label').values, 'val_y': val_df['label'].values,
                'test_x': test_df.drop(columns='label').values, 'test_y': test_df['label'].values
            }

    def build(self, rsna_root, chexpert_root, verbose=False):
        self._prepare_mnist()
        self._prepare_rsna(rsna_root)
        self._prepare_chexpert(chexpert_root)

        pca_data = np.vstack(self.pca_pool)
        # PCA fitting nutzt IMMER den CALIB_SEED für konsistenten Feature-Space
        pca = PCA(n_components=self.n_components, random_state=self.calib_seed).fit(pca_data)
        scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(pca_data))

        if verbose:
            print("\n" + "="*95)
            print(f"{'FEDERATED DATA ORCHESTRATION SUMMARY (FIXED PCA)':^95}")
            print("="*95)
            print(f"PCA Calibration Pool: {len(pca_data)} samples (Balanced & Seed-Fixed)")
            for src, count in self.pca_sources.items(): print(f"  ├─ {src}: {count} images")
            print("-" * 95)

        final_loaders = {}
        for sid, d in self.raw_pools.items():
            def make_l(x, y, shuf=False):
                x_pca = pca.transform(x.astype(np.float32))
                x_scaled = scaler.transform(x_pca)
                return DataLoader(
                    PneumoniaDataset(torch.tensor(x_scaled, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32).view(-1, 1)),
                    batch_size=self.batch_size, shuffle=shuf, drop_last=True
                )

            final_loaders[sid] = {
                'train': make_l(d['train_x'], d['train_y'], True),
                'val': make_l(d['val_x'], d['val_y']),
                'test': make_l(d['test_x'], d['test_y'])
            }
            self.metadata[sid] = {'minority_idx': np.argmin(np.bincount(d['train_y'].astype(int)))}

        return final_loaders, self.metadata

def get_federated_pca_loaders(rsna_root, chexpert_root, current_seed=42, verbose=False):
    orchestrator = FederatedDataOrchestrator(n_components=config.NUM_FEATURES,
                                             batch_size=config.BATCH_SIZE,
                                             experiment_seed=current_seed)
    return orchestrator.build(rsna_root, chexpert_root, verbose=verbose)

def get_centralized_loader(all_fed_loaders, batch_size):
    all_x = torch.cat([s['train'].dataset.dataset for s in all_fed_loaders.values()])
    all_y = torch.cat([s['train'].dataset.labels for s in all_fed_loaders.values()])
    return DataLoader(PneumoniaDataset(all_x, all_y), batch_size=batch_size, shuffle=True, drop_last=True)