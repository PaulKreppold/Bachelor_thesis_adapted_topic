import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import os
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import medmnist

# =============================================================================
# 1. KONFIGURATION
# =============================================================================
MY_SEEDS = [42, 117, 52, 31, 10]
CALIBRATION_SEED = 42
BASE_DIR = "ablation_results_final"
os.makedirs(BASE_DIR, exist_ok=True)


# =============================================================================
# 2. DATA PREPARATION (PRÄZISE DIMENSIONEN)
# =============================================================================
def get_scientific_data(dataset_name, mode, n_qubits, subset_size=1000):
    rng = np.random.RandomState(CALIBRATION_SEED)

    if dataset_name == "MNIST":
        from torchvision import datasets
        raw = datasets.MNIST(root='./data', train=True, download=True)
        idx = (raw.targets == 0) | (raw.targets == 1)
        x_raw = raw.data[idx].float().numpy() / 255.0
        y = raw.targets[idx].float().numpy()
    else:
        info = medmnist.INFO['pneumoniamnist']
        DataClass = getattr(medmnist, info['python_class'])
        ds = DataClass(split='train', download=True)
        x_raw = ds.imgs.astype(float) / 255.0
        y = ds.labels.flatten().astype(float)

    indices = rng.permutation(len(x_raw))[:subset_size]
    x_sub = x_raw[indices].reshape(len(indices), -1)
    y_sub = y[indices]

    if mode == "PCA":
        # DENSE ANGLE: Benötigt 2 Features pro Qubit (RY + RZ)
        target_dim = 2 * n_qubits
        pca = PCA(n_components=target_dim, random_state=CALIBRATION_SEED)
        x_p = pca.fit_transform(x_sub)
        x_p = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(x_p)
        print(f"--- PCA Mode: {x_p.shape[1]} Features (Dense Angle) ---")
    else:
        # AMPLITUDE EMBEDDING: Benötigt 2^n Amplituden
        target_dim = 2 ** n_qubits
        if x_sub.shape[1] > target_dim:
            pca = PCA(n_components=target_dim, random_state=CALIBRATION_SEED)
            x_p = pca.fit_transform(x_sub)
        else:
            pad_size = target_dim - x_sub.shape[1]
            x_p = np.pad(x_sub, ((0, 0), (0, pad_size)), mode='constant')

        norm = np.linalg.norm(x_p, axis=1, keepdims=True)
        x_p = x_p / (norm + 1e-12)
        print(f"--- AE Mode: {x_p.shape[1]} Amplituden ---")

    full_ds = TensorDataset(
        torch.tensor(x_p).float(),
        torch.tensor(y_sub).float().view(-1, 1)
    )

    t_size = int(0.8 * subset_size)
    train_ds, val_ds = random_split(
        full_ds, [t_size, subset_size - t_size],
        generator=torch.Generator().manual_seed(CALIBRATION_SEED)
    )
    return train_ds, val_ds


# =============================================================================
# 3. VQC MODELL (STRUKTUR AUS BASE_LINE.PY ÜBERNOMMEN)
# =============================================================================
class AblationVQC(nn.Module):
    def __init__(self, n_qubits, n_layers, encoding, ansatz, measure):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.measure = measure

        # Device wie in Base_Line.py
        self.dev = qml.device("default.qubit", wires=n_qubits)

        if ansatz == "StronglyEntangling":
            # weight_shapes wie in Base_Line.py
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        else:
            weight_shapes = {"weights": (n_layers, n_qubits)}

        def circuit(inputs, weights):
            # ENCODING
            if self.encoding == "AE":
                qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.0)
            else:
                # Dense Angle Encoding wie in Base_Line.py
                for q in range(self.n_qubits):
                    qml.RY(inputs[q], wires=q)
                    qml.RZ(inputs[q + self.n_qubits], wires=q)

            # ANSATZ
            if ansatz == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                for l in range(self.n_layers):
                    for q in range(self.n_qubits):
                        qml.RY(weights[l, q], wires=q)
                    for q in range(self.n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

            # MESSUNG: Liste von expvals wie in Base_Line.py
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # QNode-Konstruktion wie in Base_Line.py
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier-Initialisierung wie in Base_Line.py
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Batch-Processing mit Fallback aus Base_Line.py
        try:
            q_out = self.qlayer(x)
        except Exception:
            # Einzeln verarbeiten falls Batch fehlschlägt
            q_out = torch.stack([self.qlayer(sample) for sample in x])

        # Aggregation und Normalisierung
        if self.measure == "mean":
            # (batch_size, n_qubits) -> (batch_size, 1)
            q_out_mean = torch.mean(q_out, dim=1)
            return ((q_out_mean + 1) / 2).view(-1, 1)
        else:
            # Softmax-Logik für "soft" Szenarien (Sanity MNIST/Baseline)
            # Wir nutzen die ersten zwei Qubits als Logits
            return F.softmax(q_out[:, :2], dim=1)[:, 1].view(-1, 1)


# =============================================================================
# 4. RUNNER
# =============================================================================
def run_matrix():
    MATRIX = [
        {"ds": "MNIST", "q": 10, "enc": "AE", "ans": "HE", "l": 15, "m": "soft", "lbl": "1_Suenkel_Sanity_MNIST"},
        {"ds": "Pneumonia", "q": 10, "enc": "AE", "ans": "HE", "l": 15, "m": "soft", "lbl": "2_Suenkel_Baseline_AE"},
        {"ds": "Pneumonia", "q": 10, "enc": "AE", "ans": "StronglyEntangling", "l": 12, "m": "mean",
         "lbl": "3_AE_MaxExpressive_12L"},
        {"ds": "Pneumonia", "q": 4, "enc": "AE", "ans": "HE", "l": 4, "m": "mean", "lbl": "4_AE_InfoLoss_4Q"},
        {"ds": "Pneumonia", "q": 4, "enc": "PCA", "ans": "StronglyEntangling", "l": 4, "m": "mean",
         "lbl": "5_PCA_Baseline_4Q"},
        {"ds": "Pneumonia", "q": 10, "enc": "PCA", "ans": "StronglyEntangling", "l": 4, "m": "mean",
         "lbl": "6_PCA_Final_Target"}
    ]

    for conf in MATRIX:
        print(f"\n🚀 Szenario: {conf['lbl']}")
        scenario_dir = os.path.join(BASE_DIR, conf["lbl"])
        os.makedirs(scenario_dir, exist_ok=True)

        train_ds, val_ds = get_scientific_data(conf["ds"], conf["enc"], conf["q"])

        for seed in MY_SEEDS:
            print(f"   🔁 Seed {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = AblationVQC(conf["q"], conf["l"], conf["enc"], conf["ans"], conf["m"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.BCELoss()

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=16)
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            for epoch in range(25):
                model.train()
                tl, tc, tt = 0.0, 0, 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    tl += loss.item()
                    tc += ((preds > 0.5) == yb).sum().item()
                    tt += len(yb)

                model.eval()
                vl, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        vl += loss.item()
                        vc += ((preds > 0.5) == yb).sum().item()
                        vt += len(yb)

                history["train_loss"].append(tl / len(train_loader))
                history["train_acc"].append(tc / tt)
                history["val_loss"].append(vl / len(val_loader))
                history["val_acc"].append(vc / vt)

            # Speichern der Ergebnisse
            torch.save(model.state_dict(), os.path.join(scenario_dir, f"seed_{seed}_model.pt"))
            with open(os.path.join(scenario_dir, f"seed_{seed}.json"), "w") as f:
                json.dump({"config": conf, "metrics": history}, f, indent=4)
            print(f"   ✅ Seed {seed} fertig.")


if __name__ == "__main__":
    run_matrix()