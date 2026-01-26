import torch
import os
import time
import numpy as np
from Base_Line import QuantumModel  # Stelle sicher, dass die Datei so heißt
from data_prep import get_pneumonia_mnist_loaders, get_pca_angle_loaders
from train_and_eval import train_model, evaluate_model, save_experiment_results

# Liste der zu untersuchenden Phasen
MODES = ["amplitude", "angle_pca"]

for MODE in MODES:
    if MODE == "amplitude":
        import config_amplitude as cfg

        ENCODINGS_TO_RUN = ["amplitude"]
    else:
        import config_angle_pca as cfg

        ENCODINGS_TO_RUN = ["angle_normal", "angle_dense"]

    print(f"\n{'#' * 60}\n# STARTE VORSTUDIE: {MODE.upper()}\n{'#' * 60}")

    # Gesamtanzahl der Durchläufe berechnen
    total_runs = len(ENCODINGS_TO_RUN) * len(cfg.QUBIT_LIST) * len(cfg.LAYER_LIST) * len(cfg.ANSATZ_LIST) * len(
        cfg.SEEDS)
    current_run = 0
    start_time_mode = time.time()

    for enc in ENCODINGS_TO_RUN:
        for q in cfg.QUBIT_LIST:
            for l in cfg.LAYER_LIST:
                for ansatz in cfg.ANSATZ_LIST:

                    # 1. Daten laden (passend zum Encoding)
                    if enc == "amplitude":
                        train_l, val_l, test_l = get_pneumonia_mnist_loaders(cfg.BATCH_SIZE, q)
                    else:
                        train_l, val_l, test_l = get_pca_angle_loaders(cfg.BATCH_SIZE, q, enc)

                    for seed in cfg.SEEDS:
                        current_run += 1

                        # Dateinamen generieren (berücksichtigt unterschiedliche Signaturen der Configs)
                        if MODE == "angle_pca":
                            file_name = cfg.get_file_name(enc, ansatz, q, l, seed)
                        else:
                            file_name = cfg.get_file_name(ansatz, q, l, seed)

                        print(f"\n[Run {current_run}/{total_runs}] -> {file_name}")

                        # Zeitmessung für diesen spezifischen Run
                        run_start_time = time.time()

                        # Modell-Setup
                        torch.manual_seed(seed)
                        model = QuantumModel(
                            num_qubits=q,
                            num_layers=l,
                            encoding=enc,
                            ansatz=ansatz
                        ).to(cfg.DEVICE)

                        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
                        criterion = torch.nn.BCELoss()

                        # 2. Training
                        history = train_model(model, train_l, val_l, optimizer, criterion, cfg.NUM_EPOCHS, cfg.DEVICE)

                        # 3. Test-Evaluation
                        t_loss, t_acc, t_cm = evaluate_model(model, test_l, criterion, cfg.DEVICE)

                        run_duration = time.time() - run_start_time

                        # 4. Ergebnisse sichern
                        test_metrics = {'loss': t_loss, 'acc': t_acc, 'cm': t_cm}
                        save_experiment_results(
                            cfg.STATS_DIR,
                            cfg.MODELS_DIR,
                            file_name,
                            history,
                            test_metrics,
                            model,
                            duration=run_duration
                        )

                        print(f"Abgeschlossen in {run_duration:.1f}s | Test-Acc: {t_acc:.4f}")

    total_duration = time.time() - start_time_mode
    print(f"\n{'=' * 60}\nPhase {MODE} beendet. Gesamtdauer: {total_duration / 60:.2f} Minuten.\n{'=' * 60}")

