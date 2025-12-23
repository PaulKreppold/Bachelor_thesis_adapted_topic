import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
from PIL import Image
from sklearn.model_selection import train_test_split
from config import batch_size, num_subs_per_client


class ApplyTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if isinstance(y, np.ndarray): y = y.item()
        if self.transform: x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class GenericPool(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert('L')
        return img, self.labels[idx]


class RSNA_Lazy_Pool(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return img, self.labels[idx]


class DataPoolManager:
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.idx0 = np.where(labels == 0)[0]
        self.idx1 = np.where(labels == 1)[0]
        np.random.seed(42)
        np.random.shuffle(self.idx0)
        np.random.shuffle(self.idx1)
        self.p0, self.p1 = 0, 0

    def draw_subset(self, size, r0):
        n0, n1 = int(size * r0), size - int(size * r0)
        if self.p0 + n0 > len(self.idx0) or self.p1 + n1 > len(self.idx1):
            raise ValueError("Pool leer!")
        sel = np.concatenate([self.idx0[self.p0:self.p0 + n0], self.idx1[self.p1:self.p1 + n1]])
        self.p0 += n0
        self.p1 += n1
        np.random.shuffle(sel)
        return sel


def create_stratified_loaders(indices, pool, labels, b_size, tr_trans, ev_trans):
    train_idx, temp_idx = train_test_split(indices, test_size=0.20, stratify=labels[indices], random_state=42)
    val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=42)

    tl = DataLoader(ApplyTransform(Subset(pool, train_idx), tr_trans), batch_size=b_size, shuffle=True)
    vl = DataLoader(ApplyTransform(Subset(pool, val_idx), ev_trans), batch_size=b_size, shuffle=False)
    tsl = DataLoader(ApplyTransform(Subset(pool, te_idx), ev_trans), batch_size=b_size, shuffle=False)
    return tl, vl, tsl


def get_all_client_loaders(batch_size=batch_size):
    client_loaders = {}
    base_dir = "/Users/paulkreppold/Local_datasets/RSNA_Pneumonia_Detection_Challenge"
    img_dir = os.path.join(base_dir, "Training/Images")
    csv_path = os.path.join(base_dir, "stage2_train_metadata.csv")

    c1_t = [transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])] * 2
    rs_t = [transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])] * 2

    # Pneumonia Silo
    m_tr, m_va, m_te = PneumoniaMNIST(split='train', download=True), PneumoniaMNIST(split='val', download=True), PneumoniaMNIST(split='test', download=True)
    m_imgs = np.concatenate([m_tr.imgs, m_va.imgs, m_te.imgs], axis=0)
    m_lbls = np.concatenate([m_tr.labels, m_va.labels, m_te.labels], axis=0).flatten()
    man_m = DataPoolManager(m_imgs, m_lbls)

    # RSNA Silo main clients 2 & 3
    df = pd.read_csv(csv_path).drop_duplicates(subset=['patientId'])
    p_a, p_b = train_test_split(df['patientId'].values, test_size=0.5, random_state=42)

    man_r2 = DataPoolManager([os.path.join(img_dir, f"{p}.png") for p in p_a],
                             df[df['patientId'].isin(p_a)]['Target'].values)
    man_r3 = DataPoolManager([os.path.join(img_dir, f"{p}.png") for p in p_b],
                             df[df['patientId'].isin(p_b)]['Target'].values)

    ratios = {
        "client_1": [0.05, 0.12, 0.18, 0.25],  # Silo 1: Pneumonie dominiert
        "client_2": [0.65, 0.75, 0.85, 0.95],  # Silo 2: Normal dominiert
        "client_3": [0.35, 0.42, 0.48, 0.55]  # Silo 3: Mix aus Pneumonie und Normal
    }

    SUB_SIZE = 1000
    cfgs = [("client_1", man_m, GenericPool(man_m.imgs, man_m.labels), ratios["client_1"], c1_t),
            ("client_2", man_r2, RSNA_Lazy_Pool(man_r2.imgs, man_r2.labels), ratios["client_2"], rs_t),
            ("client_3", man_r3, RSNA_Lazy_Pool(man_r3.imgs, man_r3.labels), ratios["client_3"], rs_t)]

    for s_id, man, pool, rs, trans in cfgs:
        for i in range(num_subs_per_client):
            indices = man.draw_subset(SUB_SIZE, rs[i])
            tl, vl, tsl = create_stratified_loaders(indices, pool, man.labels, batch_size, trans[0], trans[1])
            client_loaders[f"{s_id}_sub_{i}"] = {"train": tl, "val": vl, "test": tsl}
    return client_loaders


def print_class_distributions(client_loaders):
    print("\n" + "=" * 90 + f"\n{'SZENARIO B: GRADIENTEN-VERTEILUNG AKTIV':^90}\n" + "=" * 90)
    for cid, loaders in client_loaders.items():
        y = [y.item() for _, lbls in loaders['train'] for y in lbls]
        print(f"{cid}: Normal {y.count(0)} ({y.count(0) / len(y):.1%}) | Pneumonie {y.count(1)}")