import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class PneumoniaDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.dataset = x_tensor
        self.labels = y_tensor

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx): return self.dataset[idx], self.labels[idx]


def get_pneumonia_mnist_loaders(batch_size=32, num_qubits=8, seed=42):
    """ Padding auf 2^num_qubits für Amplitude Embedding mit Seed-Handling """
    dim = 2 ** num_qubits
    side = int(np.sqrt(dim))
    transform = transforms.Compose([
        transforms.Resize((side, side)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    data_flag = 'pneumoniamnist'
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])

    train_ds = DataClass(split='train', transform=transform, download=True)
    val_ds = DataClass(split='val', transform=transform, download=True)
    test_ds = DataClass(split='test', transform=transform, download=True)

    # Generator für reproduzierbaren Shuffle im DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    def load(ds, shuffle=True):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=True, generator=g if shuffle else None)

    return load(train_ds), load(val_ds, False), load(test_ds, False)


def get_pca_angle_loaders(batch_size=32, num_qubits=10, encoding_type="angle_dense", seed=42):
    """ PCA mit fixem Feature-Seed (42) und variablem DataLoader-Seed """
    n_comp = 2 * num_qubits if encoding_type == "angle_dense" else num_qubits

    data_flag = 'pneumoniamnist'
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    datasets = {s: DataClass(split=s, transform=transform, download=True) for s in ['train', 'val', 'test']}

    def get_xy(ds):
        x = np.stack([item[0].numpy() for item in ds])
        y = np.array([item[1] for item in ds])
        return x, y

    x_train, y_train = get_xy(datasets['train'])
    x_val, y_val = get_xy(datasets['val'])
    x_test, y_test = get_xy(datasets['test'])

    # PCA Seed fixieren (42), damit Features über Modell-Seeds hinweg identisch sind
    pca = PCA(n_components=n_comp, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Generator für Shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    def create(x, y, shuffle=False):
        ds = PneumoniaDataset(torch.tensor(x, dtype=torch.float32),
                              torch.tensor(y, dtype=torch.float32).view(-1, 1))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=True, generator=g if shuffle else None)

    return create(x_train, y_train, True), create(x_val, y_val), create(x_test, y_test)

