import glob
import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from medmnist import PneumoniaMNIST

# Globale Pfaddefinitionen
Pneumonia_Datasets_PATH = '/Users/paulkreppold/Library/Mobile Documents/com~apple~CloudDocs/bachelor thesis/Pneumonia_datasets'
RSNA_BASE_PATH = '/Users/paulkreppold/Library/Mobile Documents/com~apple~CloudDocs/bachelor thesis/RSNA_Pneumonia_Detection_Challenge'
EXTERNAL_DATA_PATH = '/Users/paulkreppold/Library/Mobile Documents/com~apple~CloudDocs/bachelor thesis/Covid19-Pneumonia-Normal Chest X-Ray Images Dataset_mendely_data'


class RSNADataset(Dataset):
    def __init__(self, images_dir, metadata_csv, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # 1. CSV laden
        df = pd.read_csv(metadata_csv)

        # Debugging: Zeige uns kurz, welche Spalten da sind, falls Fehler auftreten
        print(f"[DEBUG] Spalten in CSV: {list(df.columns)}")

        # 2. Die Spalte mit dem Klassennamen finden
        # Deine Zeile: "id, ..., 0, No Lung Opacity / Not Normal, ..."
        # Wir suchen die Spalte, die diesen Text enthält. Meist heißt sie 'class'.
        class_col = None
        if 'class' in df.columns:
            class_col = 'class'
        elif 'Class' in df.columns:
            class_col = 'Class'
        else:
            # Fallback: Wir suchen die Spalte, die den Text enthält
            for col in df.columns:
                # Checke den ersten Eintrag (oder einen beliebigen), ob er String ist
                if df[col].dtype == object:
                    if df[col].str.contains('Lung Opacity').any():
                        class_col = col
                        break

        if class_col is None:
            raise KeyError("Konnte die Spalte mit 'Normal' / 'Lung Opacity' nicht automatisch finden.")

        print(f"[INFO] Nutze Spalte '{class_col}' zum Filtern.")
        print(f"[RSNA RAW] Gesamtanzahl Bilder: {len(df)}")

        # 3. FILTERN (Der entscheidende Schritt)
        # Wir behalten nur: 'Normal' UND 'Lung Opacity'
        # Wir löschen: 'No Lung Opacity / Not Normal'
        df_clean = df[df[class_col].isin(['Normal', 'Lung Opacity'])].copy()

        # Duplikate entfernen (falls mehrere Boxen pro Bild gelistet sind, brauchen wir das Bild trotzdem nur einmal)
        df_clean = df_clean.drop_duplicates(subset=['patientId'])

        print(f"[RSNA CLEANED] Nur Normal/Pneumonia: {len(df_clean)}")
        print(f"               Entfernt (Andere Erkrankungen): {len(df) - len(df_clean)}")

        # 4. Daten zuweisen
        self.image_ids = df_clean['patientId'].values

        # Wir setzen die Labels basierend auf dem Text, um 100% sicher zu sein
        # Normal -> 0
        # Lung Opacity -> 1
        self.labels = df_clean[class_col].apply(lambda x: 1 if x == 'Lung Opacity' else 0).values.astype(np.int64)

        # Optionaler Pfad-Check (kann man auskommentieren, wenn es zu lange dauert)
        self.valid_indices = []
        for i, pid in enumerate(self.image_ids):
            # Prüfen auf .png (da du sagtest, du hast den Ordner "Images" konvertiert)
            if os.path.exists(os.path.join(images_dir, f"{pid}.png")):
                self.valid_indices.append(i)

        self.image_ids = self.image_ids[self.valid_indices]
        self.labels = self.labels[self.valid_indices]

        # Verteilung anzeigen
        neg = np.sum(self.labels == 0)
        pos = np.sum(self.labels == 1)
        print(f"[FINAL] Valid Dataset: {len(self.image_ids)} Bilder. (Gesund: {neg}, Pneumonie: {pos})")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.images_dir, f"{img_id}.png")

        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, label


class RSNAPreprocessingTransform:
    def __init__(self, output_size=32, crop_factor=0.8):
        self.output_size = output_size
        # crop_factor 0.8 bedeutet: Wir behalten die mittleren 80% des Bildes
        self.crop_factor = crop_factor
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        # PIL → numpy
        img = np.array(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # 1. CLAHE (Kontrast verbessern, solange die Auflösung noch hoch ist)
        img = self.clahe.apply(img)

        # 2. NEU: CENTER CROP (Ränder abschneiden)
        h, w = img.shape
        crop_h = int(h * self.crop_factor)
        crop_w = int(w * self.crop_factor)

        # Startpunkte berechnen
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2

        # Zuschneiden
        img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 3. Resize (WICHTIG: Interpolation beachten)
        # cv2.INTER_AREA ist mathematisch besser beim Verkleinern (Downsampling)
        # als der Standard (INTER_LINEAR), da es Aliasing vermeidet.
        img = cv2.resize(img, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

        # zurück zu PIL
        img = Image.fromarray(img)

        return img


class ImagePathDataset(Dataset):
    """Einfaches Dataset für Mendeley-Daten"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def custom_collate_fn(batch):
    batch = [(img, lbl) for img, lbl in batch if lbl is not None]
    if len(batch) == 0:
        return None
    images, labels = zip(*batch)
    images = torch.stack(images)

    # 🔧 Hier wird jedes Label auf einen Skalar reduziert
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


# === NEUE HILFSFUNKTION FÜR EXTERNEN DATENSATZ ===
def create_external_loader(base_path, transform_test, custom_collate_fn, num_samples_per_class=1000, seed=42):
    """
    Sammelt Bildpfade von PNEUMONIA/NORMAL, zieht eine zufällige Stichprobe
    und erstellt den finalen DataLoader.
    """
    random.seed(seed)

    pneumonia_path = os.path.join(base_path, 'PNEUMONIA')
    normal_path = os.path.join(base_path, 'NORMAL')

    if not os.path.exists(pneumonia_path) or not os.path.exists(normal_path):
        raise FileNotFoundError(f"Mindestens einer der externen Pfade existiert nicht: {base_path}")

    # 1. Pfade sammeln und Labels zuweisen (0=Normal, 1=Pneumonia)
    # Annahme: Alle Bilder sind .png oder .jpeg

    pneumonia_files = glob.glob(os.path.join(pneumonia_path, '*.*'))
    normal_files = glob.glob(os.path.join(normal_path, '*.*'))

    print(f"[INFO] Externe Daten gefunden: PNEUMONIA={len(pneumonia_files)}, NORMAL={len(normal_files)}")

    # 2. Zufällige Stichprobe ziehen
    if len(pneumonia_files) >= num_samples_per_class:
        pneu_subset = random.sample(pneumonia_files, num_samples_per_class)
    else:
        print(f"[WARNUNG] Nur {len(pneumonia_files)} PNEUMONIA-Bilder verfügbar. Alle werden verwendet.")
        pneu_subset = pneumonia_files

    if len(normal_files) >= num_samples_per_class:
        normal_subset = random.sample(normal_files, num_samples_per_class)
    else:
        print(f"[WARNUNG] Nur {len(normal_files)} NORMAL-Bilder verfügbar. Alle werden verwendet.")
        normal_subset = normal_files

    # 3. Pfade und Labels zusammenfassen
    image_paths = pneu_subset + normal_subset
    labels = [1] * len(pneu_subset) + [0] * len(normal_subset)

    # 4. Dataset und DataLoader erstellen
    external_ds = ImagePathDataset(image_paths, labels, transform=transform_test)

    external_loader = DataLoader(
        external_ds,
        batch_size=64,
        shuffle=False,  # Kein Mischen für Test- oder Evaluations-Loader
        collate_fn=custom_collate_fn
    )

    print(
        f"[INFO] External Test Loader erstellt mit {len(external_ds)} Bildern (Target 0: {len(normal_subset)}, Target 1: {len(pneu_subset)})")

    return external_loader


# NEUE, KORRIGIERTE SIGNATUR
def split_client_dataset(base_name, base_dataset, all_client_fractions, custom_collate_fn, seed=42):
    """
    Teilt ein Dataset in vier disjunkte Subclients auf, basierend auf all_client_fractions[base_name].
    """
    np.random.seed(seed)
    total_len = len(base_dataset)
    # HIER WIRD AUS DEM GESAMTEN DICTIONARY die FRACTION GEHOLT:
    fraction = all_client_fractions.get(base_name, 0.1)
    used_len = int(total_len * fraction)

    indices = np.random.choice(total_len, used_len, replace=False)
    subset = Subset(base_dataset, indices)

    sub_len = used_len // 4
    sub_lengths = [sub_len] * 4
    sub_lengths[-1] += used_len - sum(sub_lengths)

    # Korrektur der Aufteilung, um 80/10/10 zu nutzen (basierend auf Ihrer letzten Anfrage)
    sub_datasets = random_split(subset, sub_lengths, generator=torch.Generator().manual_seed(seed))

    loaders = {}
    for i, sub_ds in enumerate(sub_datasets, 1):
        # ANGEPASSTE SPLIT-VERTEILUNG: 80% / 10% / 10%
        n_train = int(0.80 * len(sub_ds))
        n_val = int(0.10 * len(sub_ds))
        n_test = len(sub_ds) - n_train - n_val

        train_ds, val_ds, test_ds = random_split(sub_ds, [n_train, n_val, n_test],
                                                 generator=torch.Generator().manual_seed(seed + i))
        cid = f"{base_name}_sub{i}"
        loaders[cid] = {
            # custom_collate_fn ist jetzt definiert
            "train": DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate_fn),
            "val": DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=custom_collate_fn),
            "test": DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=custom_collate_fn),
        }
    return loaders


# === HAUPTFUNKTION ===
def load_client_data(client_qubits, custom_collate_fn, client_fractions=None, seed=42):
    """
    Lädt Daten für alle Clients und Subclients.
    - MedMNIST -> client_1
    - RSNA -> client_2, client_3
    - Kaggle -> client_4
    - Subclients erhalten disjunkte Datensätze nach client_fractions
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    client_train_loaders, client_val_loaders, client_test_loaders = {}, {}, {}

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

    rsna_preprocess = RSNAPreprocessingTransform(output_size=32)

    rsna_transform_train = transforms.Compose([
        rsna_preprocess,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


    # ===== CLIENT 1: MedMNIST =====
    if any(k.startswith('client_1') for k in client_qubits):
        train_ds_full = PneumoniaMNIST(split='train', transform=common_transform_train, download=True)
        val_ds_full = PneumoniaMNIST(split='val', transform=common_transform_test, download=True)
        test_ds_full = PneumoniaMNIST(split='test', transform=common_transform_test, download=True)

        subclients = [c for c in client_qubits if c.startswith('client_1')]
        num_subclients = len(subclients)

        def disjunctive_split_to_loaders(base_dataset, is_train_set=False):
            total_len = len(base_dataset)
            sub_len = total_len // num_subclients
            sub_lengths = [sub_len] * num_subclients
            sub_lengths[-1] += total_len - sum(sub_lengths)

            sub_datasets = random_split(base_dataset, sub_lengths,
                                        generator=torch.Generator().manual_seed(seed))

            loaders_dict = {}
            batch_size = 32 if is_train_set else 64

            for i, sub_ds in enumerate(sub_datasets):
                cid = f"{base_name}_sub{i + 1}"
                if cid in client_qubits:
                    loaders_dict[cid] = DataLoader(sub_ds,
                                                   batch_size=batch_size,
                                                   shuffle=is_train_set,
                                                   collate_fn=custom_collate_fn)
            return loaders_dict

        base_name = 'client_1'
        train_loaders_split = disjunctive_split_to_loaders(train_ds_full, is_train_set=True)
        val_loaders_split = disjunctive_split_to_loaders(val_ds_full, is_train_set=False)
        test_loaders_split = disjunctive_split_to_loaders(test_ds_full, is_train_set=False)

        for cid in subclients:
            if cid in client_qubits:
                client_train_loaders[cid] = train_loaders_split[cid]
                client_val_loaders[cid] = val_loaders_split[cid]
                client_test_loaders[cid] = test_loaders_split[cid]

                n_train = len(client_train_loaders[cid].dataset)
                n_val = len(client_val_loaders[cid].dataset)
                n_test = len(client_test_loaders[cid].dataset)

                # --- Labelverarbeitung robust ---
                train_subset = client_train_loaders[cid].dataset
                labels_sub = np.array([int(train_ds_full.labels[i]) for i in train_subset.indices], dtype=int)

                print(f"[INFO] {cid}: Train={n_train}, Val={n_val}, Test={n_test} | Class dist={np.bincount(labels_sub)}")

    # ===== CLIENT 2 & 3: RSNA =====
    if any(k.startswith('client_2') or k.startswith('client_3') for k in client_qubits):
        rsna_images = os.path.join(RSNA_BASE_PATH, 'Training', 'Images')
        # Das ist die Datei aus deinem Screenshot, die schon alles enthält:
        rsna_csv = os.path.join(RSNA_BASE_PATH, 'stage2_train_metadata.csv')

        rsna_full = RSNADataset(
            images_dir=rsna_images,
            metadata_csv=rsna_csv,
            # KEIN class_info_csv mehr nötig!
            transform=rsna_transform_train
        )

        total_per_base = 5000
        labels = np.array(rsna_full.labels)
        idx0, idx1 = np.where(labels == 0)[0], np.where(labels == 1)[0]
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)

        client2_idx = np.concatenate([idx0[:total_per_base // 2], idx1[:total_per_base // 2]])
        client3_idx = np.concatenate([idx0[total_per_base // 2:total_per_base], idx1[total_per_base // 2:total_per_base]])

        base_dict = {'client_2': client2_idx, 'client_3': client3_idx}

        for base_client, base_idx in base_dict.items():
            sub_split = split_client_dataset(
                base_client,
                Subset(rsna_full, base_idx),
                client_fractions,
                custom_collate_fn,
                seed
            )

            for sub_id, loaders_dict in sub_split.items():
                client_train_loaders[sub_id] = loaders_dict["train"]
                client_val_loaders[sub_id] = loaders_dict["val"]
                client_test_loaders[sub_id] = loaders_dict["test"]

                train_subset = client_train_loaders[sub_id].dataset
                train_indices = train_subset.indices
                labels_sub = np.array([int(rsna_full[i][1]) for i in train_indices], dtype=int)

                n_train = len(loaders_dict["train"].dataset)
                n_val = len(loaders_dict["val"].dataset)
                n_test = len(loaders_dict["test"].dataset)

                print(f"[INFO] {sub_id}: Train={n_train}, Val={n_val}, Test={n_test} | Class dist={np.bincount(labels_sub)}")

    # ===== CLIENT 4: Kaggle =====
    if any(k.startswith('client_4') for k in client_qubits):
        train_path = os.path.join(Pneumonia_Datasets_PATH, "train_data/client_4/train")
        test_path = os.path.join(Pneumonia_Datasets_PATH, "test_data/client_4/test")
        train_full = datasets.ImageFolder(root=train_path, transform=common_transform_train)
        test_full = datasets.ImageFolder(root=test_path, transform=common_transform_test)

        total_len_train = len(train_full)
        train_indices = np.arange(total_len_train)
        np.random.shuffle(train_indices)
        shuffled_train_full = Subset(train_full, train_indices)

        sub_split = split_client_dataset(
            'client_4',
            shuffled_train_full,
            client_fractions,
            custom_collate_fn,
            seed
        )


        subclients = [c for c in client_qubits if c.startswith("client_4")]
        num_subclients = len(subclients)

        # split test set into num_subclients parts
        test_indices = np.arange(len(test_full))
        np.random.shuffle(test_indices)
        test_split = np.array_split(test_indices, num_subclients)

        for i, cid in enumerate(subclients):

            # TRAIN-LOADER werden bei dir bereits über split_client_dataset erzeugt
            client_train_loaders[cid] = sub_split[cid]["train"]
            client_val_loaders[cid] = sub_split[cid]["val"]

            # TEST-SET KORREKT ZUORDNEN
            client_test_loaders[cid] = DataLoader(
                Subset(test_full, test_split[i]),
                batch_size=64,
                shuffle=False,
                collate_fn=custom_collate_fn
            )

            train_subset = client_train_loaders[cid].dataset
            original_indices = [train_indices[i] for i in train_subset.indices]
            labels_sub = np.array([int(train_full.targets[i]) for i in original_indices], dtype=int)

            n_train = len(client_train_loaders[cid].dataset)
            n_val = len(client_val_loaders[cid].dataset)
            n_test = len(client_test_loaders[cid].dataset)

            print(f"[INFO] {cid}: Train={n_train}, Val={n_val}, Test={n_test} | Class dist={np.bincount(labels_sub)}")

    # ===== EXTERNE DATEN =====
    try:
        external_loader = create_external_loader(
            base_path=EXTERNAL_DATA_PATH,
            transform_test=common_transform_test,
            custom_collate_fn=custom_collate_fn,
            num_samples_per_class=1000,
            seed=seed
        )
    except FileNotFoundError as e:
        print(f"[FEHLER] Externe Daten konnten nicht geladen werden: {e}")
        external_loader = None

    return client_train_loaders, client_val_loaders, client_test_loaders, external_loader
