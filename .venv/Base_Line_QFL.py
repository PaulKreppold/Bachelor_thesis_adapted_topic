import torch
import torch.nn as nn
import numpy as np
import random
import pennylane as qml

# Eigene Module
from Base_Line import QuantumModel
from data_prep import load_data_pca
from train_and_eval import train_client, evaluate_model
from plots import plot_train_accuracy_and_loss

# Config Import
from config import device, client_qubits

# ==========================================
# KONFIGURATION
# ==========================================
SEEDS = [42, 1337, 2024, 112, 19]
num_epochs = 50
learning_rate = 0.001
batch_size = 4
num_layers = 4

# Fest auf client_1 gesetzt
client_id = "client_1"

# Qubits direkt holen (Crash bei Fehler, kein Fallback)
num_qubits = client_qubits[client_id]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Start Baseline Training auf Device: {device}")
    print(f"Client: {client_id} | Qubits: {num_qubits} | Layers: {num_layers}")

    # 1. Daten laden
    print("\n[INIT] Lade Daten...")
    train_loaders, val_loaders, test_loaders = load_data_pca(batch_size=batch_size)

    # Zugriff auf die Loader von client_1
    train_loader = train_loaders[client_id]
    val_loader_dict = {client_id: val_loaders[client_id]}  # Für Validierung während Training
    test_loader = test_loaders[client_id]

    # ==========================================
    # SEED LOOP
    # ==========================================
    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f" START TRAINING RUN - SEED {seed} ")
        print("=" * 60)

        set_seed(seed)

        # 2. Modell initialisieren
        print(f"[Seed {seed}] Initialisiere Quantum Model...")
        model = QuantumModel(num_qubits=num_qubits, num_layers=num_layers).to(device)

        # 3. Setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 4. Training
        print(f"[Seed {seed}] Starte Training ({num_epochs} Epochen)...")
        results = train_client(
            model=model,
            client_train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            client_id=client_id,
            num_epochs=num_epochs,
            client_val_loaders=val_loader_dict,
            print_last_only=False,
            last_round=True
        )

        # 5. Evaluation
        print(f"[Seed {seed}] Evaluiere auf Test-Set...")
        eval_out = evaluate_model(model, test_loader, criterion)

        (test_acc, test_loss, test_f1, test_auc, test_auc_pr,
         test_precision, test_recall, test_cm,
         test_probs, test_labels, test_class_f1) = eval_out

        # 6. Ergebnisse
        print("-" * 40)
        print(f"ERGEBNISSE SEED {seed}")
        print("-" * 40)
        print(f"Test Accuracy:  {test_acc:.2f}%")
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test F1-Score:  {test_f1:.4f}")
        print(f"Test AUC:       {test_auc:.4f}")
        print(f"Confusion Matrix:\n{test_cm}")
        print("-" * 40)

        # 7. Plotten (ohne Speichern)
        plot_train_accuracy_and_loss(
            train_losses=results["train_losses"],
            train_accuracies=results["train_accuracies"],
            seed=seed
        )

    print("\n" + "=" * 60)
    print(" ALLE SEEDS ABGESCHLOSSEN ")
    print("=" * 60)