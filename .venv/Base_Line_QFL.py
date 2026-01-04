import torch
import os
import numpy as np
import medmnist
from medmnist import INFO
import config
from data_prep import get_pca_data_loaders
from train_and_eval import train_model, evaluate_model
from Base_Line import QuantumModel
from plots import (
    plot_averaged_results,
    plot_test_accuracy_distribution,
    plot_prediction_histogram,
    log_sample_predictions,
    visualize_top_errors,
    plot_master_comparison
)


def main():
    # 0. Verzeichnisse vorbereiten
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Hier speichern wir die Daten aller Experimente für den finalen Master-Plot
    master_data = []
    overall_summary = []

    # 1. Roh-Daten einmalig laden (wichtig für die Bild-Visualisierung der Fehler)
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    raw_test_dataset = DataClass(split='test', download=True)

    # 2. Daten für alle Experimente laden (PCA-Komponenten bleiben gleich)
    print(f"\nLade Daten mit PCA: {config.NUM_FEATURES} Komponenten...")
    train_loader, val_loader, test_loader = get_pca_data_loaders(
        batch_size=config.BATCH_SIZE,
        n_components=config.NUM_FEATURES
    )

    # --- EXPERIMENT LOOP (Matrix aus config.py) ---
    for exp_cfg in config.EXPERIMENT_CONFIGS:
        enc = exp_cfg["encoding"]
        use_scale = exp_cfg["use_scaling"]

        exp_name = f"Enc-{enc}_Scale-{use_scale}"
        exp_dir = os.path.join(config.RESULTS_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"STARTE EXPERIMENT: {exp_name}")
        print(f"{'#' * 60}\n")

        all_histories = []
        test_results = []

        # --- SEED LOOP für statistische Absicherung ---
        for i, seed in enumerate(config.SEEDS):
            print(f"--- [Lauf {i + 1}/{len(config.SEEDS)} | Seed: {seed}] ---")

            # Reproduzierbarkeit sicherstellen
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Modell initialisieren
            model = QuantumModel(
                num_qubits=config.NUM_QUBITS,
                num_layers=config.NUM_LAYERS,
                encoding=enc,
                use_scaling=use_scale
            ).to(config.DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

            # Scheduler: Reduziert LR wenn der Val-Loss stagniert
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
            )

            criterion = torch.nn.BCEWithLogitsLoss()

            # --- TRAINING ---
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=config.NUM_EPOCHS,
                seed=seed,
                device=config.DEVICE
            )
            all_histories.append(history)

            # --- EVALUATION ---
            test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
            test_results.append({'loss': test_loss, 'acc': test_acc, 'cm': test_cm})

            # --- LOGGING & PLOTS PRO SEED ---
            # 1. Histogramm der VQC-Ausgaben
            hist_path = os.path.join(exp_dir, f"seed_{seed}_hist.png")
            plot_prediction_histogram(model, test_loader, hist_path, device=config.DEVICE)

            # 2. Text-Log der Predictions (Basis für die Top-Fehler Analyse)
            sample_log_path = os.path.join(exp_dir, f"seed_{seed}_samples.txt")
            log_sample_predictions(model, test_loader, sample_log_path, device=config.DEVICE)

            # 3. Top-Fehler Bilder visualisieren
            error_img_path = os.path.join(exp_dir, f"seed_{seed}_top_errors.png")
            visualize_top_errors(sample_log_path, raw_test_dataset, num_samples=3, save_path=error_img_path)

            print(f"Seed {seed} fertig. Test-Acc: {test_acc * 100:.2f}%")

        # --- AUSWERTUNG PRO EXPERIMENT ---
        f_test_accs = [r['acc'] for r in test_results]
        f_test_losses = [r['loss'] for r in test_results]

        res_entry = {
            'name': exp_name,
            'avg_acc': np.mean(f_test_accs),
            'std_acc': np.std(f_test_accs),
            'avg_loss': np.mean(f_test_losses),
            'histories': all_histories  # Für den Master-Plot
        }
        overall_summary.append(res_entry)
        master_data.append(res_entry)

        # Plots für das aktuelle Experiment (über alle Seeds gemittelt)
        plot_averaged_results(all_histories, exp_dir)
        plot_test_accuracy_distribution(f_test_accs, exp_dir)

        print(f"\nAbgeschlossen: {exp_name}")
        print(f"Mean Accuracy: {res_entry['avg_acc'] * 100:.2f}% ± {res_entry['std_acc'] * 100:.2f}%")

    # --- FINALE GESAMT-AUSWERTUNG ---
    # 1. Text-Zusammenfassung (Tabelle)
    summary_path = os.path.join(config.RESULTS_DIR, "OVERALL_COMPARISON.txt")
    with open(summary_path, "w") as f:
        f.write("FINALE AUSWERTUNG: VQC ENCODING & SCALING VERGLEICH\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Experiment':35} | {'Test Accuracy':18} | {'Test Loss':10}\n")
        f.write("-" * 70 + "\n")
        for r in overall_summary:
            line = f"{r['name']:35} | {r['avg_acc'] * 100:6.2f}% ± {r['std_acc'] * 100:5.2f}% | {r['avg_loss']:.4f}\n"
            f.write(line)

    # 2. DER MASTER-PLOT (Alle 4 Experimente in einer Grafik)
    print("\nErstelle Master-Vergleichs-Plot...")
    plot_master_comparison(master_data, config.RESULTS_DIR)

    print(f"\n{'=' * 60}")
    print(f"ALLE EXPERIMENTE ABGESCHLOSSEN.")
    print(f"Ergebnisse gespeichert in: {config.RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()