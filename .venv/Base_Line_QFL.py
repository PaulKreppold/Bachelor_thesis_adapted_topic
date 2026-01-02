import torch
import os
import numpy as np
import medmnist
from medmnist import INFO
import config  # SEEDS, BATCH_SIZE, NUM_QUBITS, NUM_LAYERS, LR, etc.
from data_prep import get_pca_data_loaders
from train_and_eval import train_model, evaluate_model
from Base_Line import QuantumModel
from plots import (
    plot_averaged_results,
    plot_test_accuracy_distribution,
    plot_prediction_histogram,
    log_sample_predictions,
    visualize_top_errors  # Stellen Sie sicher, dass dies in plots.py ist
)


def main():
    # 0. Vorbereitung & Verzeichnisse
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ablation_inits = ["identity", "uniform"]

    # 1. Daten laden
    # PCA-Loader für das Training
    train_loader, val_loader, test_loader = get_pca_data_loaders(
        batch_size=config.BATCH_SIZE,
        n_components=config.NUM_FEATURES
    )

    # Roh-Daten für die Visualisierung der Röntgenbilder (ohne PCA/Transform)
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    raw_test_dataset = DataClass(split='test', download=True)

    # 2. Schleife über Initialisierungen (Ablation)
    for init_mode in ablation_inits:
        ablation_name = f"init_{init_mode}"
        ablation_dir = os.path.join(config.RESULTS_DIR, ablation_name)
        os.makedirs(ablation_dir, exist_ok=True)

        summary_file_path = os.path.join(ablation_dir, "final_summary.txt")

        print(f"\n{'=' * 60}\nStarte Ablation: {ablation_name}\n{'=' * 60}\n")

        all_histories = []
        test_results = []

        # 3. Schleife über Seeds
        for i, seed in enumerate(config.SEEDS):
            print(f"--- [SEED {i + 1}/{len(config.SEEDS)}: {seed}] ---")

            # Reproduzierbarkeit sicherstellen
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Modell initialisieren mit dem init_mode aus deiner Klasse
            model = QuantumModel(
                num_qubits=config.NUM_QUBITS,
                num_layers=config.NUM_LAYERS,
                init_mode=init_mode
            ).to(config.DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )

            # MSE Loss wie besprochen
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training
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

            # Test-Evaluation (Wichtig: evaluate_model sollte nun MSELoss nutzen)
            test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
            test_results.append({
                'loss': test_loss,
                'acc': test_acc,
                'cm': test_cm
            })

            # Histogramm der Predictions speichern
            hist_file = os.path.join(ablation_dir, f"seed_{seed}_pred_hist.png")
            plot_prediction_histogram(model, test_loader, hist_file, device=config.DEVICE)

            # Detailliertes Logging (Individueller Loss pro Sample)
            sample_file = os.path.join(ablation_dir, f"seed_{seed}_sample_outputs.txt")
            log_sample_predictions(model, test_loader, sample_file, device=config.DEVICE, num_batches=5)

            # Visualisierung der Top-Fehler (Worst Samples) für diesen Seed
            error_plot_path = os.path.join(ablation_dir, f"seed_{seed}_top_errors.png")
            visualize_top_errors(sample_file, raw_test_dataset, num_samples=3, save_path=error_plot_path)

            print(f"[Seed {seed}] Test-Acc: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}")

        # 4. Statistische Auswertung über alle Seeds
        f_train_acc = [h['acc'][-1] for h in all_histories]
        f_val_acc = [h['val_acc'][-1] for h in all_histories]
        f_test_acc = [r['acc'] for r in test_results]
        f_test_loss = [r['loss'] for r in test_results]

        all_test_cms = np.array([r['cm'] for r in test_results])
        mean_cm = np.mean(all_test_cms, axis=0)

        def calc_stats(data):
            return np.mean(data), np.std(data)

        stats = {
            "Train Acc": calc_stats(f_train_acc),
            "Val Acc": calc_stats(f_val_acc),
            "Test Loss": calc_stats(f_test_loss),
            "Test Acc": calc_stats(f_test_acc),
        }

        # Zusammenfassung formatieren
        output_str = (
            f"{'=' * 75}\n"
            f"FINALE ERGEBNISSE: {ablation_name.upper()} (Mittelwert ± StdAbw)\n"
            f"{'=' * 75}\n"
            f"Train Accuracy: {stats['Train Acc'][0] * 100:.2f}% ± {stats['Train Acc'][1] * 100:.2f}%\n"
            f"Val Accuracy:   {stats['Val Acc'][0] * 100:.2f}% ± {stats['Val Acc'][1] * 100:.2f}%\n"
            f"{'-' * 75}\n"
            f"TEST LOSS:      {stats['Test Loss'][0]:.4f} ± {stats['Test Loss'][1]:.4f}\n"
            f"TEST ACCURACY:  {stats['Test Acc'][0] * 100:.2f}% ± {stats['Test Acc'][1] * 100:.2f}%\n\n"
            f"DURCHSCHNITTLICHE TEST CONFUSION MATRIX:\n"
            f"   TN: {mean_cm[0, 0]:6.1f} | FP: {mean_cm[0, 1]:6.1f}\n"
            f"   FN: {mean_cm[1, 0]:6.1f} | TP: {mean_cm[1, 1]:6.1f}\n"
            f"{'=' * 75}\n"
        )

        print(output_str)
        with open(summary_file_path, "w") as f:
            f.write(output_str)

        # Übergreifende Plots pro Ablation
        plot_test_accuracy_distribution(f_test_acc, ablation_dir)
        plot_averaged_results(all_histories, ablation_dir)

    print(f"\nProzess abgeschlossen. Alle Ergebnisse unter: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()