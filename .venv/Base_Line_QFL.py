import torch
import os
import numpy as np
import config
from data_prep import get_pca_data_loaders
from train_and_eval import train_model, evaluate_model
from Base_Line import QuantumModel
import plots


def main():
    # 1. Daten laden (einmalig für alle Experimente)
    # n_components entspricht NUM_FEATURES in config (z.B. 20 für 10 Qubits Dense Encoding)
    print(f"Lade MedMNIST mit PCA (n={config.NUM_FEATURES})...")
    train_loader, val_loader, test_loader = get_pca_data_loaders(
        batch_size=config.BATCH_SIZE,
        n_components=config.NUM_FEATURES
    )

    # --- EXPERIMENT LOOP 1: Layer-Anzahl (4, 6, 8, 10, 12) ---
    for layers in config.layer_configs:

        # --- EXPERIMENT LOOP 2: Initialisierung (Xavier, Kaiming, Small Normal, Uniform) ---
        for init_name in config.init_methods:

            exp_name = f"Layers_{layers}_Init_{init_name}"
            exp_dir = os.path.join(config.RESULTS_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            print(f"\n" + "=" * 60)
            print(f"STARTE EXPERIMENT: {exp_name.upper()}")
            print("=" * 60)

            all_histories = []
            all_metrics = []

            # --- SEED LOOP: Statistische Absicherung ---
            for seed in config.SEEDS:
                print(f"\n>>> Seed: {seed}")
                seed_dir = os.path.join(exp_dir, f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)

                # Reproduzierbarkeit sicherstellen
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Modell mit dynamischen Parametern instanziieren
                model = QuantumModel(n_layers=layers, init_method=init_name).to(config.DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=8
                )
                criterion = torch.nn.BCELoss()

                # --- TRAINING ---
                history = train_model(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, scheduler=scheduler,
                    epochs=config.NUM_EPOCHS, seed=seed, device=config.DEVICE
                )
                all_histories.append(history)

                # --- EVALUATION ---
                loss, acc, cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
                all_metrics.append({'loss': loss, 'acc': acc, 'cm': cm})

                # Lokale Speicherung der Konfidenz-Verteilung pro Seed
                plots.plot_prediction_histogram(
                    model, test_loader, os.path.join(seed_dir, "final_hist.png"), config.DEVICE
                )

            # --- AUSWERTUNG PRO EXPERIMENT (Mittelung über alle Seeds) ---
            print(f"\nAggregiere Ergebnisse für {exp_name}...")

            # 1. Lokale Grafiken für diesen spezifischen Versuchsaufbau
            plots.plot_averaged_results(all_histories, exp_dir)
            plots.plot_average_confusion_matrix(all_metrics, exp_name, exp_dir)
            plots.plot_test_accuracy_distribution([m['acc'] for m in all_metrics], exp_dir)
            plots.save_final_statistics(all_metrics, exp_dir)

            # 2. WICHTIG: Rohdaten speichern für den globalen Vergleich am Ende
            plots.save_raw_results(all_histories, all_metrics, exp_dir)

    # --- FINALE GLOBALE ANALYSE ---
    # Nachdem alle Kombinationen durchgelaufen sind, erstellen wir die großen Vergleichsplots
    print(f"\n" + "#" * 60)
    print("ERSTELLE GLOBALE VERGLEICHS-GRAFIKEN ÜBER ALLE VARIANTEN")
    print("#" * 60)

    # Erstellt das globale Liniendiagramm (Train Loss) und den großen Test-Acc Boxplot
    plots.plot_global_comparison(
        config.RESULTS_DIR, config.layer_configs, config.init_methods
    )

    # Erstellt das Grid aus Confusion Matrices zum Vergleich der Fehlermuster
    plots.plot_global_cm_grid(
        config.RESULTS_DIR, config.layer_configs, config.init_methods
    )

    print(f"\nAlle Experimente und Vergleiche abgeschlossen.")
    print(f"Die Ergebnisse befinden sich in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()