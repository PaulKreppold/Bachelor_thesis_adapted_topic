import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class PneumoniaDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]