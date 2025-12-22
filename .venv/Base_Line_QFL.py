import torch
import torch.optim as optim
import numpy as np
import random
import os

# Import der Konfiguration aus config.py
from config import *

# Import der eigenen Module
from Base_Line import QuantumModel
from train_and_eval import train_client, evaluate_model, visualize_batch_analysis
from plots import plot_train_accuracy_and_loss_comparative
from data_prep import get_all_client_loaders, print_class_distributions


def set_seed(seed):
    """Fixiert alle Random-Seeds für maximale Vergleichbarkeit."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    header_width = 100
    print("\n" + "█" * header_width)
    print(f"{' VQC ARCHITECTURE COMPARISON: 4-WAY ANALYSIS ':^{header_width}}")
    print(f"{' StronglyEntangling vs. EfficientSU2 | Pure vs. Calibrated ':^{header_width}}")
    print("█" * header_width)

    # 1. Daten laden
    print("[*] Initialisiere Data-Loaders...")
    client_loaders = get_all_client_loaders(batch_size=BATCH_SIZE)

    # Übersicht der Klassenverteilung pro Client anzeigen
    print_class_distributions(client_loaders)

    # Definition der Architekturen aus dem Paper-Kontext
    ansatz_types = ["StronglyEntangling", "EfficientSU2"]

    # 2. Hauptschleife über die Seeds
    for seed in SEEDS:
        print(f"\n\n{'#' * 100}")
        print(f"#{f' STARTING EXPERIMENTS FOR SEED {seed} '.center(98)}#")
        print(f"{'#' * 100}")

        for client_id in clients:
            # 3. Architektur-Schleife (VQC Typen)
            for ansatz in ansatz_types:

                # 4. Kalibrierungs-Schleife (Pure vs. Scale & Bias)
                for use_scale in [False, True]:

                    # WICHTIG: Seed vor JEDEM Durchlauf neu fixieren für identische Initialgewichte
                    set_seed(seed)

                    # Label für Logs und Plots erstellen
                    calib_status = "Calibrated" if use_scale else "Pure"
                    mode_label = f"{ansatz}_{calib_status}"

                    print(f"\n>>> [CLIENT: {client_id}] | ARCH: {ansatz} | MODE: {calib_status} | Seed: {seed}")

                    # Modell erstellen mit dynamischer Architektur-Wahl
                    model = QuantumModel(
                        num_qubits=10,
                        num_layers=NUM_LAYERS,
                        use_scaling=use_scale,
                        ansatz_type=ansatz
                    ).to(device)

                    # Optimizer (LR kommt aus config.py)
                    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                    # --- TRAINING ---
                    results = train_client(
                        model=model,
                        train_loader=client_loaders[client_id]['train'],
                        optimizer=optimizer,
                        client_id=client_id,
                        num_epochs=NUM_EPOCHS,
                        device=device,
                        mode_label=mode_label
                    )

                    # --- EVALUATION ---
                    print(f"[*] Evaluiere {client_id} auf Test-Set...")
                    acc, (f1_normal, f1_krank), _, _ = evaluate_model(
                        model,
                        client_loaders[client_id]['test'],
                        device=device
                    )

                    # Detaillierte Resultat-Ausgabe
                    print(f"\n[RESULTAT - {mode_label}]")
                    print(f"Acc: {acc:.2f}% | F1-Normal: {f1_normal:.4f} | F1-Krank: {f1_krank:.4f}")

                    # --- BATCH-ANALYSE (Visualisierung der Rohwerte und Parameter) ---
                    visualize_batch_analysis(
                        model=model,
                        test_loader=client_loaders[client_id]['test'],
                        client_id=client_id,
                        mode_label=mode_label,
                        device=device
                    )

                    # --- PLOTTING ---
                    try:
                        plot_train_accuracy_and_loss_comparative(
                            train_losses=results["train_losses"],
                            train_accuracies=results["train_accuracies"],
                            seed=seed,
                            client_id=client_id,
                            mode_label=mode_label
                        )
                    except Exception as e:
                        print(f"[!] Plot-Fehler bei {client_id} ({mode_label}): {e}")

    print("\n" + "█" * header_width)
    print(f"{' ALLE EXPERIMENTE (4 KONFIGURATIONEN) ERFOLGREICH BEENDET ':^{header_width}}")
    print("█" * header_width + "\n")


if __name__ == "__main__":
    main()