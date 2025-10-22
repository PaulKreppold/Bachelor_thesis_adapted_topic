import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from medmnist import PneumoniaMNIST
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def custom_collate_fn(batch):
    batch = [(img, lbl) for img, lbl in batch if lbl is not None]
    if len(batch) == 0:
        return None
    images, labels = zip(*batch)
    images = torch.stack([torch.as_tensor(img) for img in images])
    labels = torch.stack([torch.as_tensor(lbl).view(1) for lbl in labels])
    return images, labels


def stratified_subset(dataset, fraction, targets):
    if fraction >= 1.0:
        return dataset

    indices = []
    targets_np = np.array(targets)

    for label in np.unique(targets_np):
        label_indices = np.where(targets_np == label)[0]
        n_keep = max(1, int(len(label_indices) * fraction))
        keep = np.random.choice(label_indices, n_keep, replace=False)
        indices.extend(keep)

    return Subset(dataset, indices)


def get_subset_targets(dataset):

    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, 'targets'):
        targets = dataset.dataset.targets
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, 'labels'):
        targets = dataset.dataset.labels
    else:
        return np.array([])

    targets_np = np.array(targets).flatten()

    if isinstance(dataset, Subset):
        indices = dataset.indices
        return targets_np[indices]

    return targets_np


def print_distribution(targets, name):

    if not isinstance(targets, np.ndarray) and not isinstance(targets, list) or len(targets) == 0:
        print(f"  {name} - WARNUNG: Keine Daten gefunden oder Labels nicht extrahierbar.")
        return

    targets_np = np.array(targets).flatten()
    total = len(targets_np)
    pneumonia_count = np.sum(targets_np == 1)
    normal_count = total - pneumonia_count

    print(f"  --- {name} Verteilung ({total} Bilder) ---")
    print(f"  Klasse 0 (Normal): {normal_count} ({normal_count / total:.2%})")
    print(f"  Klasse 1 (Pneumonie): {pneumonia_count} ({pneumonia_count / total:.2%})")



def load_client_data(client_qubits, fraction=1.0):
    client_train_loaders = {}
    client_val_loaders = {}
    client_test_loaders = {}

    for client, num_qubits in client_qubits.items():

        if isinstance(fraction, dict):
            frac = fraction.get(client, 1.0)
        else:
            frac = fraction

        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        if client == 'client_1':
            train_dataset = PneumoniaMNIST(split='train', transform=transform_train, download=True)
            val_dataset = PneumoniaMNIST(split='val', transform=transform_test, download=True)
            test_dataset = PneumoniaMNIST(split='test', transform=transform_test, download=True)

            train_targets_full = train_dataset.labels
            val_targets_full = val_dataset.labels
            test_targets_full = test_dataset.labels

            train_dataset = stratified_subset(train_dataset, frac, train_targets_full)
            val_dataset = stratified_subset(val_dataset, frac, val_targets_full)
            test_dataset = stratified_subset(test_dataset, frac, test_targets_full)

            train_targets_final = get_subset_targets(train_dataset)
            val_targets_final = get_subset_targets(val_dataset)
            test_targets_final = get_subset_targets(test_dataset)


        elif client == 'client_3':
            train_dataset_path = f'/Users/paulkreppold/Downloads/Pneumonia_datasets/train_data/{client}/train'
            full_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform_train)
            full_targets = full_dataset.targets

            pneumonia_indices = [i for i, label in enumerate(full_targets) if label == 1]
            normal_indices = [i for i, label in enumerate(full_targets) if label == 0]

            pneumonia_train, pneumonia_temp = train_test_split(pneumonia_indices, test_size=0.2, random_state=42)
            pneumonia_val, pneumonia_test = train_test_split(pneumonia_temp, test_size=0.5, random_state=42)

            normal_train, normal_temp = train_test_split(normal_indices, test_size=0.2, random_state=42)
            normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)

            train_indices = pneumonia_train + normal_train
            val_indices = pneumonia_val + normal_val
            test_indices = pneumonia_test + normal_test

            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)

            train_dataset.targets = [full_targets[i] for i in train_indices]
            val_dataset.targets = [full_targets[i] for i in val_indices]
            test_dataset.targets = [full_targets[i] for i in test_indices]

            train_dataset = stratified_subset(train_dataset, frac, train_dataset.targets)
            val_dataset = stratified_subset(val_dataset, frac, val_dataset.targets)
            test_dataset = stratified_subset(test_dataset, frac, test_dataset.targets)

            train_targets_final = get_subset_targets(train_dataset)
            val_targets_final = get_subset_targets(val_dataset)
            test_targets_final = get_subset_targets(test_dataset)


        else:
            train_dataset_path = f'/Users/paulkreppold/Downloads/Pneumonia_datasets/train_data/{client}/train'
            test_dataset_path = f'/Users/paulkreppold/Downloads/Pneumonia_datasets/test_data/{client}/test'

            full_train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform_train)
            full_test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform_test)
            full_targets = full_train_dataset.targets

            pneumonia_indices = [i for i, label in enumerate(full_targets) if label == 1]
            normal_indices = [i for i, label in enumerate(full_targets) if label == 0]

            pneumonia_train, pneumonia_val = train_test_split(pneumonia_indices, test_size=0.12, random_state=42)
            normal_train, normal_val = train_test_split(normal_indices, test_size=0.12, random_state=42)

            train_indices = pneumonia_train + normal_train
            val_indices = pneumonia_val + normal_val

            train_dataset = Subset(full_train_dataset, train_indices)
            val_dataset = Subset(full_train_dataset, val_indices)
            test_dataset = full_test_dataset

            train_dataset.targets = [full_targets[i] for i in train_indices]
            val_dataset.targets = [full_targets[i] for i in val_indices]

            train_dataset = stratified_subset(train_dataset, frac, train_dataset.targets)
            val_dataset = stratified_subset(val_dataset, frac, val_dataset.targets)
            test_dataset = stratified_subset(test_dataset, frac, full_test_dataset.targets)

            train_targets_final = get_subset_targets(train_dataset)
            val_targets_final = get_subset_targets(val_dataset)
            test_targets_final = get_subset_targets(test_dataset)

        client_train_loaders[client] = DataLoader(train_dataset, batch_size=16, shuffle=True)
        client_val_loaders[client] = DataLoader(val_dataset, batch_size=64, shuffle=False)
        client_test_loaders[client] = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print(f"{client} - fraction={frac:.2f} -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print_distribution(train_targets_final, "Train")
        print_distribution(val_targets_final, "Val")
        print_distribution(test_targets_final, "Test")
        print("-" * 40)

    return client_train_loaders, client_val_loaders, client_test_loaders