import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ----------------------------
# Dataset-Klasse für PCA/AngleEncoding
# ----------------------------
class PneumoniaDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


# ----------------------------
# AmplitudeEmbedding Loader
# ----------------------------
def get_pneumonia_mnist_loaders(batch_size=32):
    """
    Lädt PneumoniaMNIST für AmplitudeEmbedding (16x16 = 256 Features)
    """
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((16, 16)),  # 16x16 = 256 Features
        transforms.ToTensor(),
    ])

    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset   = DataClass(split='val',   transform=transform, download=True)
    test_dataset  = DataClass(split='test',  transform=transform, download=True)

    def loader(dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader(train_dataset), loader(val_dataset), loader(test_dataset)


# ----------------------------
# PCA + Dense AngleEncoding Loader
# ----------------------------
def get_pca_angle_loaders(batch_size=32, n_components=20, seed=42):
    """
    Lädt PneumoniaMNIST, wendet PCA an und skaliert Komponenten auf [0, π].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Standardgröße für PCA
        transforms.ToTensor(),
    ])

    # Bilder laden
    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset   = DataClass(split='val',   transform=transform, download=True)
    test_dataset  = DataClass(split='test',  transform=transform, download=True)

    # Hilfsfunktion: Dataset → flacher Tensor
    def dataset_to_tensor(dataset):
        imgs = []
        labels = []
        for img, label in dataset:
            imgs.append(img.view(-1).numpy())  # flatten
            labels.append(label)
        return np.stack(imgs), np.array(labels)

    x_train, y_train = dataset_to_tensor(train_dataset)
    x_val,   y_val   = dataset_to_tensor(val_dataset)
    x_test,  y_test  = dataset_to_tensor(test_dataset)

    # PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca   = pca.transform(x_val)
    x_test_pca  = pca.transform(x_test)

    # Scaling auf [0, π] für Dense Angle Encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_train_scaled = scaler.fit_transform(x_train_pca)
    x_val_scaled   = scaler.transform(x_val_pca)
    x_test_scaled  = scaler.transform(x_test_pca)

    # PyTorch Datasets & Loader
    train_loader = DataLoader(
        PneumoniaDataset(torch.tensor(x_train_scaled, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        PneumoniaDataset(torch.tensor(x_val_scaled, dtype=torch.float32),
                         torch.tensor(y_val, dtype=torch.float32).view(-1, 1)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        PneumoniaDataset(torch.tensor(x_test_scaled, dtype=torch.float32),
                         torch.tensor(y_test, dtype=torch.float32).view(-1, 1)),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


