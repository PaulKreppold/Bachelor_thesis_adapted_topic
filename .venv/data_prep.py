import os
import torch
import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ======================================================
# KONFIGURATION & PFADE
# ======================================================
BASE_DIR = '/Users/paulkreppold/Local_datasets'
RSNA_PATH = os.path.join(BASE_DIR, 'RSNA_Pneumonia_Detection_Challenge')
CHEXPERT_PATH = os.path.join(BASE_DIR, 'CheXpert_Frontal_Processed')
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset_mendely_data')

# ======================================================
# LABEL CACHE & HELPER
# ======================================================
_LABEL_CACHE = {}


def get_dataset_id(dataset):
    return id(dataset)


def cache_labels(dataset, labels):
    _LABEL_CACHE[get_dataset_id(dataset)] = labels


def get_cached_labels(dataset):
    return _LABEL_CACHE.get(get_dataset_id(dataset))


def get_labels_from_dataset(dataset: Dataset, use_cache: bool = True) -> List[int]:
    """Optimierte Label-Extraktion mit Caching."""
    if use_cache:
        cached = get_cached_labels(dataset)
        if cached is not None: return cached

    if isinstance(dataset, Subset):
        full_labels = get_labels_from_dataset(dataset.dataset, use_cache=use_cache)
        result = [full_labels[i] for i in dataset.indices]
    elif isinstance(dataset, TransformWrapper):
        result = get_labels_from_dataset(dataset.dataset, use_cache=use_cache)
    elif isinstance(dataset, ConcatDataset):
        result = []
        for ds in dataset.datasets:
            result.extend(get_labels_from_dataset(ds, use_cache=use_cache))
    elif hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
        if isinstance(labels, (np.ndarray, list)):
            result = [int(l.item() if isinstance(l, (np.ndarray, np.generic)) else l) for l in labels]
        else:
            result = [int(labels)]
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
        if isinstance(labels, np.ndarray):
            result = [int(l) for l in labels.flatten()]
        elif isinstance(labels, list):
            result = [int(l) for l in labels]
        else:
            result = [int(labels)]
    else:
        # Fallback
        result = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                if item: result.append(int(item[1]))
            except:
                continue

    if use_cache: cache_labels(dataset, result)
    return result


def analyze_source_dataset(name: str, dataset: Dataset):
    """Zeigt Statistiken der Quelle vor dem Split an."""
    print(f"\n--- ANALYSE QUELLE: {name} ---")
    try:
        labels = get_labels_from_dataset(dataset, use_cache=True)
        total = len(labels)
        n_pos = sum(labels)
        n_neg = total - n_pos
        ratio = n_pos / total if total > 0 else 0
        print(f"Total Images: {total}")
        print(f"Pneumonia (1): {n_pos} ({ratio * 100:.1f}%)")
        print(f"Normal (0):    {n_neg} ({(1 - ratio) * 100:.1f}%)")
    except Exception as e:
        print(f"Konnte Stats nicht laden: {e}")
    print("-" * 40)


def stratified_split(dataset: Dataset, fractions: List[float], seed: int = 42) -> List[Subset]:
    """Robuster Stratified Split."""
    labels = np.array(get_labels_from_dataset(dataset, use_cache=True))
    n = len(labels)
    if n == 0: return [Subset(dataset, []) for _ in fractions]

    rng = np.random.default_rng(seed)
    subsets_indices = [[] for _ in fractions]

    unique_classes = np.unique(labels)
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        cum_fracs = np.cumsum(fractions)
        split_points = (cum_fracs[:-1] * len(cls_indices)).astype(int)
        splits = np.split(cls_indices, split_points)

        for i, split in enumerate(splits):
            if i < len(subsets_indices):
                subsets_indices[i].extend(split.tolist())

    for s in subsets_indices: rng.shuffle(s)

    result_subsets = []
    for indices in subsets_indices:
        sub = Subset(dataset, indices)
        cache_labels(sub, labels[indices].tolist())
        result_subsets.append(sub)

    return result_subsets


def create_imbalanced_subset(dataset: Dataset, target_size: int, ratio_pos: float, seed: int = 42) -> Subset:
    """Erstellt Subset mit exaktem Ratio (oder best-effort)."""
    labels = np.array(get_labels_from_dataset(dataset, use_cache=True))
    rng = np.random.default_rng(seed)

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    n_pos = int(target_size * ratio_pos)
    n_neg = target_size - n_pos

    # Checks
    if len(idx_pos) < n_pos: n_pos = len(idx_pos)
    if len(idx_neg) < n_neg: n_neg = len(idx_neg)

    chosen_pos = rng.choice(idx_pos, n_pos, replace=False) if n_pos > 0 else []
    chosen_neg = rng.choice(idx_neg, n_neg, replace=False) if n_neg > 0 else []

    chosen = np.concatenate([chosen_pos, chosen_neg]).astype(int)
    rng.shuffle(chosen)

    sub = Subset(dataset, chosen.tolist())
    cache_labels(sub, labels[chosen].tolist())
    return sub


def custom_collate_fn(batch):
    clean = [(x, y) for x, y in batch if x is not None]
    if not clean: return None
    imgs, lbls = zip(*clean)
    return torch.stack(imgs), torch.tensor([int(l) for l in lbls], dtype=torch.long)


# ======================================================
# DATASETS & ROBUSTER WRAPPER
# ======================================================
class TransformWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # Initiale Label-Abfrage
        self._labels = get_labels_from_dataset(dataset, use_cache=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            # 1. Check auf None
            if item is None:
                return None

            # 2. Check auf Struktur (Bild, Label)
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return None

            x, y = item

            # 3. Transform
            if self.transform:
                x = self.transform(x)

            return x, int(y)

        except Exception:
            # Fehlerhaftes Bild/Index abfangen
            return None

    def get_labels(self):
        return self._labels


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        cache_labels(self, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('L')
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
        except:
            return None

    def get_labels(self):
        return self.labels


class RSNADataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = os.path.join(root_dir, "Training", "Images")
        df = pd.read_csv(csv_file).drop_duplicates(subset=['patientId'])
        self.patient_ids = df['patientId'].tolist()
        self.labels = [int(x) for x in df['Target'].tolist()]
        cache_labels(self, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        for ext in ['.png', '.dcm', '.jpg']:
            path = os.path.join(self.root_dir, f"{pid}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert('L'), self.labels[idx]
                except:
                    pass
        return None

    def get_labels(self):
        return self.labels


class CheXpertBinaryPneumoniaDedup(Dataset):
    def __init__(self, root_dir, csv_file, strategy='frontal_only'):
        self.root_dir = root_dir
        df = pd.read_csv(csv_file).fillna(0)
        df_pos = df[df.get("Pneumonia", 0) == 1.0].copy();
        df_pos["target"] = 1
        df_neg = df[df.get("No Finding", 0) == 1.0].copy();
        df_neg["target"] = 0

        data = pd.concat([df_pos, df_neg], ignore_index=True)
        data['fname'] = data['Path'].apply(os.path.basename)
        data['pid'] = data['fname'].apply(lambda x: x.split('_')[0])

        if 'Frontal/Lateral' in data.columns:
            data = data[data['Frontal/Lateral'] == 'Frontal']

        if strategy == 'random':
            data = data.groupby('pid', as_index=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(
                drop=True)
        else:
            data = data.groupby('pid', as_index=False).first()

        self.fnames = data['fname'].tolist()
        self.labels = [int(x) for x in data['target'].tolist()]
        cache_labels(self, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            return Image.open(os.path.join(self.root_dir, self.fnames[idx])).convert('L'), self.labels[idx]
        except:
            return None

    def get_labels(self):
        return self.labels


# ======================================================
# CORE SPLIT LOGIC
# ======================================================
def split_client_dataset(base_name: str, base_dataset: Dataset,
                         transform_train, transform_test, collate_fn, seed: int = 42) -> dict:
    """
    Logik für RSNA (Client 2/3) und CheXpert (Client 4):
    - Train: Unique & Skewed.
    - Val: Shared pro Skew-Gruppe (lokaler Skew).
    - Test: Unified / Shared über ALLE Subclients eines Clients hinweg.
    """
    n_sub = 4
    sub_chunks = stratified_split(base_dataset, [1.0 / n_sub] * n_sub, seed=seed)

    loaders = {}

    all_val_parts = []
    all_test_parts = []
    train_datasets = []
    subclient_ratios = []

    TRAIN_SIZE = 1000
    VAL_TARGET_TOTAL = 520
    TEST_TARGET_TOTAL = 620

    # Standard Ratios (Train Skew)
    default_ratio = 0.5
    if "client_2" in base_name:
        default_ratio = 0.60
    elif "client_3" in base_name:
        default_ratio = 0.40
    elif "client_4" in base_name:
        default_ratio = 0.25

    for i, chunk in enumerate(sub_chunks):
        sub_id = i + 1
        curr_seed = seed + sub_id * 100

        pool_tr, pool_va, pool_te = stratified_split(chunk, [0.7, 0.15, 0.15], seed=curr_seed)

        # SKEW LOGIK
        current_ratio = default_ratio
        if "client_4" in base_name and sub_id == 1:
            current_ratio = 0.10  # Extreme Skew für C4 Sub1

        subclient_ratios.append(current_ratio)

        final_train = create_imbalanced_subset(pool_tr, TRAIN_SIZE, current_ratio, seed=curr_seed)
        train_datasets.append(final_train)

        all_val_parts.append(pool_va)
        all_test_parts.append(pool_te)

    # --- SHARED POOLS ---
    full_val_concat = ConcatDataset(all_val_parts)
    full_test_concat = ConcatDataset(all_test_parts)

    # 1. Validation Datasets (SKEW AWARE)
    val_datasets_cache = {}
    unique_ratios = set(subclient_ratios)
    for r in unique_ratios:
        val_datasets_cache[r] = create_imbalanced_subset(
            full_val_concat, VAL_TARGET_TOTAL, r, seed=seed + int(r * 100)
        )

    # 2. Test Dataset (UNIFIED / BALANCED)
    UNIFIED_TEST_RATIO = 0.40
    shared_test_ds = create_imbalanced_subset(
        full_test_concat, TEST_TARGET_TOTAL, UNIFIED_TEST_RATIO, seed=seed + 999
    )

    for i in range(n_sub):
        cid = f"{base_name}_sub{i + 1}"
        ratio = subclient_ratios[i]

        loaders[cid] = {
            "train": DataLoader(TransformWrapper(train_datasets[i], transform_train), batch_size=32, shuffle=True,
                                collate_fn=collate_fn),
            "val": DataLoader(TransformWrapper(val_datasets_cache[ratio], transform_test), batch_size=64, shuffle=False,
                              collate_fn=collate_fn),
            "test": DataLoader(TransformWrapper(shared_test_ds, transform_test), batch_size=64, shuffle=False,
                               collate_fn=collate_fn)
        }

    return loaders


# ======================================================
# EXTERNAL DATA LOADING
# ======================================================
def create_external_loader(external_path, transform, collate_fn, seed=42):
    if not os.path.exists(external_path): return None
    img_paths, labels = [], []
    for root, _, files in os.walk(external_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, f))
                label = 1 if any(k in root.lower() for k in ['pneum', 'pneu', 'covid']) else 0
                labels.append(label)
    if not img_paths: return None
    print(f"[INFO] External Dataset: {len(img_paths)} Bilder")
    ds = ImagePathDataset(img_paths, labels, transform=transform)
    cache_labels(ds, labels)
    return DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)


# ======================================================
# MAIN LOADING FUNCTION
# ======================================================
def load_client_data(client_qubits: dict, collate_fn=None, seed: int = 42, **kwargs):
    if collate_fn is None: collate_fn = custom_collate_fn

    tf_train = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomAffine(10, translate=(0.1, 0.1)), transforms.ToTensor()
    ])
    tf_test = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    loaders_train, loaders_val, loaders_test = {}, {}, {}

    # === CLIENT 1: PNEUMONIAMNIST ===
    if any(k.startswith('client_1') for k in client_qubits):
        print("\n[LOAD] Client 1 (MedMNIST)...")
        from medmnist import PneumoniaMNIST
        tr_full = PneumoniaMNIST(split='train', download=True, transform=None)
        va_full = PneumoniaMNIST(split='val', download=True, transform=None)
        te_full = PneumoniaMNIST(split='test', download=True, transform=None)

        analyze_source_dataset("MedMNIST Train", tr_full)

        # 1. Training in 4 Chunks
        tr_chunks = stratified_split(tr_full, [0.25, 0.25, 0.25, 0.25], seed=seed)

        # 2. Skewed Validation für Sub 1 vorbereiten
        val_skewed = create_imbalanced_subset(va_full, 520, 0.90, seed=seed + 50)

        # 3. Test Unified (te_full ist 624 Bilder)
        unified_test = te_full

        for i, chunk in enumerate(tr_chunks):
            sub_id = i + 1
            cid = f"client_1_sub{sub_id}"

            # --- SPLIT LOGIK ---
            if sub_id == 1:
                # == SKEWED 90/10 ==
                # Wir fordern 90% Pneu. Da der Chunk evtl. nicht genug Pneu hat,
                # nimmt create_imbalanced_subset so viel wie möglich -> ca. 973 Bilder.
                final_train = create_imbalanced_subset(chunk, 1000, ratio_pos=0.90, seed=seed + 1)
                final_val = val_skewed
            else:
                # == NORMAL / BALANCED ==
                # Wir nutzen hier AUCH create_imbalanced_subset, aber mit der nativen Ratio.
                # Das verhindert den "Rest"-Fehler beim Splitten.
                # Berechne native Ratio des Chunks:
                lbls = get_labels_from_dataset(chunk, use_cache=True)
                native_ratio = sum(lbls) / len(lbls) if len(lbls) > 0 else 0.5

                # Zwinge auf exakt 1000 Bilder (oder max available)
                final_train = create_imbalanced_subset(chunk, 1000, native_ratio, seed=seed + i)

                # Fixe Validation auf original MedMNIST Val
                final_val = va_full

            # --- ZUWEISUNG ---
            if cid in client_qubits:
                loaders_train[cid] = DataLoader(TransformWrapper(final_train, tf_train), batch_size=32, shuffle=True,
                                                collate_fn=collate_fn)
                loaders_val[cid] = DataLoader(TransformWrapper(final_val, tf_test), batch_size=64, shuffle=False,
                                              collate_fn=collate_fn)
                loaders_test[cid] = DataLoader(TransformWrapper(unified_test, tf_test), batch_size=64, shuffle=False,
                                               collate_fn=collate_fn)

    # === CLIENT 2 & 3: RSNA ===
    c2_active = any(k.startswith('client_2') for k in client_qubits)
    c3_active = any(k.startswith('client_3') for k in client_qubits)

    if (c2_active or c3_active) and os.path.exists(os.path.join(RSNA_PATH, "stage2_train_metadata.csv")):
        print("\n[LOAD] RSNA...")
        rsna_full = RSNADataset(RSNA_PATH, os.path.join(RSNA_PATH, "stage2_train_metadata.csv"))
        analyze_source_dataset("RSNA Full", rsna_full)

        if c2_active and c3_active:
            ds_c2, ds_c3 = stratified_split(rsna_full, [0.5, 0.5], seed=seed)
        elif c2_active:
            ds_c2, ds_c3 = rsna_full, None
        else:
            ds_c2, ds_c3 = None, rsna_full

        if c2_active:
            ldrs = split_client_dataset("client_2", ds_c2, tf_train, tf_test, collate_fn, seed)
            for k, v in ldrs.items():
                if k in client_qubits:
                    loaders_train[k], loaders_val[k], loaders_test[k] = v['train'], v['val'], v['test']

        if c3_active:
            ldrs = split_client_dataset("client_3", ds_c3, tf_train, tf_test, collate_fn, seed + 999)
            for k, v in ldrs.items():
                if k in client_qubits:
                    loaders_train[k], loaders_val[k], loaders_test[k] = v['train'], v['val'], v['test']

    # === CLIENT 4: CHEXPERT ===
    if any(k.startswith('client_4') for k in client_qubits) and os.path.exists(CHEXPERT_PATH):
        print("\n[LOAD] CheXpert...")
        chex_full = CheXpertBinaryPneumoniaDedup(os.path.join(CHEXPERT_PATH, "train"),
                                                 os.path.join(CHEXPERT_PATH, "train.csv"))
        analyze_source_dataset("CheXpert", chex_full)

        ldrs = split_client_dataset("client_4", chex_full, tf_train, tf_test, collate_fn, seed)
        for k, v in ldrs.items():
            if k in client_qubits:
                loaders_train[k], loaders_val[k], loaders_test[k] = v['train'], v['val'], v['test']

    ext_load = create_external_loader(EXTERNAL_DATA_PATH, tf_test, collate_fn, seed)

    return loaders_train, loaders_val, loaders_test, ext_load