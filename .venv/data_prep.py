import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
from collections import Counter

# ======================================================
# 1. PFAD-KONFIGURATION
# ======================================================
BASE_DIR = '/Users/paulkreppold/Library/Mobile Documents/com~apple~CloudDocs/bachelor thesis'

Pneumonia_Datasets_PATH = os.path.join(BASE_DIR, 'Pneumonia_datasets')
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset_mendely_data')


# ======================================================
# 2. HILFSKLASSEN
# ======================================================

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('L')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden von {self.image_paths[idx]}: {e}")
            return None, self.labels[idx]


def custom_collate_fn(batch):
    batch = [(img, lbl) for img, lbl in batch if img is not None]
    if len(batch) == 0:
        return None
    images, labels = zip(*batch)
    images = torch.stack(images)

    clean_labels = []
    for lbl in labels:
        if isinstance(lbl, torch.Tensor):
            clean_labels.append(lbl.item() if lbl.numel() == 1 else int(lbl.squeeze()[0]))
        elif isinstance(lbl, (list, np.ndarray)):
            clean_labels.append(int(lbl[0]))
        else:
            clean_labels.append(int(lbl))

    labels = torch.tensor(clean_labels, dtype=torch.long)
    return images, labels


# ======================================================
# 3. SPLIT & SKEW LOGIK (MIT SIZING)
# ======================================================

def get_labels_from_dataset(dataset):
    if isinstance(dataset, Subset):
        parent_labels = get_labels_from_dataset(dataset.dataset)
        return [parent_labels[i] for i in dataset.indices]

    if hasattr(dataset, 'targets'):
        return dataset.targets

    if hasattr(dataset, 'labels'):
        labels = dataset.labels
        if len(labels) > 0 and isinstance(labels[0], (list, np.ndarray)):
            return [int(l[0]) for l in labels]
        return labels

    labels = []
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        labels.append(int(lbl))
    return labels


def create_skewed_subset(dataset, majority_class, ratio=0.9, seed=42, target_size=None):
    """
    Erstellt ein Subset mit Skew und spezifischer Zielgröße (Upsampling/Downsampling).
    """
    labels = np.array(get_labels_from_dataset(dataset))
    indices_maj = np.where(labels == majority_class)[0]
    indices_min = np.where(labels != majority_class)[0]

    if target_size is None:
        target_size = len(dataset)

    n_maj_target = int(target_size * ratio)
    n_min_target = target_size - n_maj_target

    rng = np.random.default_rng(seed)

    # Entscheidung ob Upsampling nötig ist (replace=True)
    replace_maj = len(indices_maj) < n_maj_target
    replace_min = len(indices_min) < n_min_target

    final_indices_maj = rng.choice(indices_maj, size=n_maj_target, replace=replace_maj)
    final_indices_min = rng.choice(indices_min, size=n_min_target, replace=replace_min)

    final_indices = np.concatenate([final_indices_maj, final_indices_min])
    rng.shuffle(final_indices)

    return Subset(dataset, final_indices)


def stratified_split(dataset, fractions, seed=42):
    labels = np.array(get_labels_from_dataset(dataset))
    unique_labels = np.unique(labels)

    indices_per_class = {label: np.where(labels == label)[0] for label in unique_labels}
    split_indices = [[] for _ in range(len(fractions))]
    rng = np.random.default_rng(seed)

    for label, indices in indices_per_class.items():
        rng.shuffle(indices)
        n_class = len(indices)
        counts = [int(f * n_class) for f in fractions]

        diff = n_class - sum(counts)
        counts[0] += diff

        current = 0
        for i, count in enumerate(counts):
            split_indices[i].extend(indices[current: current + count])
            current += count

    for indices in split_indices:
        rng.shuffle(indices)

    return [Subset(dataset, indices) for indices in split_indices]


def split_client_dataset(base_name, base_dataset, all_client_fractions, custom_collate_fn, seed=42,
                         has_external_test=False):
    # --- Schritt 1: Aufteilung in 4 Subclients ---
    sub_fractions = [0.25, 0.25, 0.25, 0.25]

    fraction_total = all_client_fractions.get(base_name, 1.0)
    if fraction_total < 1.0:
        base_dataset, _ = stratified_split(base_dataset, [fraction_total, 1 - fraction_total], seed=seed)

    sub_datasets = stratified_split(base_dataset, sub_fractions, seed=seed)

    loaders = {}
    for i, sub_ds in enumerate(sub_datasets, 1):
        cid = f"{base_name}_sub{i}"

        # --- Schritt 2: Interner Split (Train/Val) ---
        if has_external_test:
            train_ds, val_ds = stratified_split(sub_ds, [0.9, 0.1], seed=seed + i)
            test_ds = None
        else:
            train_ds, val_ds, test_ds = stratified_split(sub_ds, [0.8, 0.1, 0.1], seed=seed + i)

        # --- MANIPULATIONEN (SKEW & SIZING) ---

        # Fall A: LDS Clients (Starker Skew, kleinere Größe)
        if cid == 'client_2_sub1':
            print(f"️ [LDS] {cid}: 90% NORMAL, Size ~550")
            train_ds = create_skewed_subset(train_ds, majority_class=0, ratio=0.9, seed=seed, target_size=550)
            val_ds = create_skewed_subset(val_ds, majority_class=0, ratio=0.9, seed=seed, target_size=70)

        elif cid == 'client_3_sub1':
            print(f" [LDS] {cid}: 90% PNEUMONIE, Size ~550")
            train_ds = create_skewed_subset(train_ds, majority_class=1, ratio=0.9, seed=seed, target_size=550)
            val_ds = create_skewed_subset(val_ds, majority_class=1, ratio=0.9, seed=seed, target_size=70)

        # Fall B: Normale Clients (Kein Skew, aber Downsampling auf 1000)
        else:
            # Wir wollen ca. 1000 Trainingsbilder.
            # Aktuell sind es ca. 1170-1180.
            target_train = 1000
            if len(train_ds) > target_train:
                frac = target_train / len(train_ds)
                # Wir behalten 'frac', den Rest verwerfen wir (_)
                train_ds, _ = stratified_split(train_ds, [frac, 1 - frac], seed=seed)

            # Validation proportional anpassen? Oder lassen?
            # Lassen wir bei ~130, das ist eine gute Größe für stabile Validation.
            # (Wenn du es strikt proportional willst: target_val = 125, aber der Unterschied ist marginal)

        loaders[cid] = {
            "train": DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn),
            "val": DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=custom_collate_fn),
            "test": DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=custom_collate_fn) if test_ds else None
        }
    return loaders


# ======================================================
# 4. HAUPTFUNKTION: LOAD CLIENT DATA
# ======================================================

def create_external_loader(base_path, transform_test, custom_collate_fn, num_samples_per_class=1000, seed=42):
    random.seed(seed)
    pneumonia_path = os.path.join(base_path, 'PNEUMONIA')
    normal_path = os.path.join(base_path, 'NORMAL')

    if not os.path.exists(pneumonia_path) or not os.path.exists(normal_path):
        raise FileNotFoundError(f"Externe Pfade nicht gefunden in: {base_path}")

    pneumonia_files = glob.glob(os.path.join(pneumonia_path, '*.*'))
    normal_files = glob.glob(os.path.join(normal_path, '*.*'))

    pneu_subset = random.sample(pneumonia_files, min(len(pneumonia_files), num_samples_per_class))
    normal_subset = random.sample(normal_files, min(len(normal_files), num_samples_per_class))

    image_paths = pneu_subset + normal_subset
    labels = [1] * len(pneu_subset) + [0] * len(normal_subset)

    external_ds = ImagePathDataset(image_paths, labels, transform=transform_test)
    print(
        f"[INFO] External Test Loader: {len(external_ds)} Bilder (Normal: {len(normal_subset)}, Pneu: {len(pneu_subset)})")

    return DataLoader(external_ds, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)


def load_client_data(client_qubits, custom_collate_fn, client_fractions=None, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    client_train_loaders, client_val_loaders, client_test_loaders = {}, {}, {}

    # --- TRANSFORMS ---
    common_transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    common_transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ==========================================
    # 1. CLIENT 1: MedMNIST
    # ==========================================
    if any(k.startswith('client_1') for k in client_qubits):
        print("\n[LOAD] Lade MedMNIST für Client 1...")
        from medmnist import PneumoniaMNIST

        train_ds_full = PneumoniaMNIST(split='train', transform=common_transform_train, download=True)
        val_ds_full = PneumoniaMNIST(split='val', transform=common_transform_test, download=True)
        test_ds_full = PneumoniaMNIST(split='test', transform=common_transform_test, download=True)

        subclients = [c for c in client_qubits if c.startswith('client_1')]
        num_subclients = len(subclients)
        fracs = [1.0 / num_subclients] * num_subclients

        train_subs = stratified_split(train_ds_full, fracs, seed=seed)
        val_subs = stratified_split(val_ds_full, fracs, seed=seed)
        test_subs = stratified_split(test_ds_full, fracs, seed=seed)

        for i, cid in enumerate(subclients):
            # HIER: Auch MedMNIST Subclients auf 1000 reduzieren?
            # Aktuell haben sie ~1180. Wir reduzieren sie auch, um Fair zu bleiben.

            t_ds = train_subs[i]
            target_train = 1000
            if len(t_ds) > target_train:
                frac = target_train / len(t_ds)
                t_ds, _ = stratified_split(t_ds, [frac, 1 - frac], seed=seed + i)

            client_train_loaders[cid] = DataLoader(t_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
            client_val_loaders[cid] = DataLoader(val_subs[i], batch_size=64, shuffle=False,
                                                 collate_fn=custom_collate_fn)
            client_test_loaders[cid] = DataLoader(test_subs[i], batch_size=64, shuffle=False,
                                                  collate_fn=custom_collate_fn)
            print(f"[INFO] {cid}: Train={len(client_train_loaders[cid].dataset)}")

    # ==========================================
    # 2. CLIENT 2 & 3: Folder-Struktur
    # ==========================================
    folder_clients = ['client_2', 'client_3']

    for base_client in folder_clients:
        subclients = [c for c in client_qubits if c.startswith(base_client)]
        if not subclients: continue

        print(f"\n[LOAD] Lade Daten für {base_client}...")
        train_path = os.path.join(Pneumonia_Datasets_PATH, "train_data", base_client, "train")
        test_path = os.path.join(Pneumonia_Datasets_PATH, "test_data", base_client, "test")

        if not os.path.exists(train_path):
            print(f"[WARNUNG] Trainingspfad nicht gefunden: {train_path}. Überspringe.")
            continue

        train_full = datasets.ImageFolder(root=train_path, transform=common_transform_train)

        has_external_test = False
        test_full = None

        if os.path.exists(test_path) and len(glob.glob(os.path.join(test_path, "*"))) > 0:
            try:
                test_full = datasets.ImageFolder(root=test_path, transform=common_transform_test)
                has_external_test = True
                print(f"[INFO] {base_client}: Externer Test-Ordner gefunden ({len(test_full)} Bilder).")
            except:
                print(f"[WARNUNG] Fehler beim Laden von Test-Daten für {base_client}. Nutze Split.")
        else:
            print(f"[INFO] {base_client}: Kein externer Test-Ordner. Erstelle Test-Split aus Training.")

        # Ruft split_client_dataset auf (enthält SKEW & SIZING Logik für Train/Val)
        loaders_split = split_client_dataset(
            base_client,
            train_full,
            client_fractions if client_fractions else {},
            custom_collate_fn,
            seed,
            has_external_test=has_external_test
        )

        test_subsets_base = []
        if has_external_test and test_full:
            num_subs = 4
            t_fracs = [1.0 / num_subs] * num_subs
            test_subsets_base = stratified_split(test_full, t_fracs, seed=seed)

        for i, cid in enumerate(subclients):
            if cid in loaders_split:
                client_train_loaders[cid] = loaders_split[cid]["train"]
                client_val_loaders[cid] = loaders_split[cid]["val"]

                if has_external_test:
                    if i < len(test_subsets_base):
                        client_test_loaders[cid] = DataLoader(test_subsets_base[i], batch_size=64, shuffle=False,
                                                              collate_fn=custom_collate_fn)
                    else:
                        print(f"[WARNUNG] Kein Test-Split mehr für {cid}.")
                else:
                    client_test_loaders[cid] = loaders_split[cid]["test"]

                # Quick Stat Check
                try:
                    labels_t = get_labels_from_dataset(client_train_loaders[cid].dataset)
                    dist_t = dict(Counter(labels_t))
                    ratio_t = dist_t.get(1, 0) / (dist_t.get(0, 1) + 1e-6)
                    print(f"[INFO] {cid}: Train={len(labels_t)} (Pneu/Norm Ratio: {ratio_t:.2f})")
                except:
                    print(f"[INFO] {cid}: Train={len(client_train_loaders[cid].dataset)}")

    try:
        print("\n[LOAD] Lade externe Mendeley Daten...")
        external_loader = create_external_loader(
            base_path=EXTERNAL_DATA_PATH,
            transform_test=common_transform_test,
            custom_collate_fn=custom_collate_fn,
            num_samples_per_class=1000,
            seed=seed
        )
    except Exception as e:
        print(f"[FEHLER] Externe Daten konnten nicht geladen werden: {e}")
        external_loader = None

    return client_train_loaders, client_val_loaders, client_test_loaders, external_loader