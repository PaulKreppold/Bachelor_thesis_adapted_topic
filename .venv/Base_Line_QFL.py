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
    visualize_top_errors
)


def main():
    # 0. Verzeichnisse vorbereiten
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Roh-Daten einmalig laden (für visualize_top_errors)
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    raw_test_dataset = DataClass(split='test', download=True)

    # --- GRID SEARCH: PROFILE (Qubits & Layers) ---
    for profile in config.GRID_PROFILES:
        n_q = profile["qubits"]
        n_l = profile["layers"]
        p_name = profile["name"]

        # WICHTIG: NUM_FEATURES ist abhängig von Qubits (2 Winkel pro Qubit)
        n_features = n_q * 2

        # 2. Daten für DIESES Profil neu laden (wegen PCA-Komponenten)
        print(
            f"\n\n{'#' * 80}\nSTARTE PROFIL: {p_name} | Qubits: {n_q} | Layer: {n_l} | Features: {n_features}\n{'#' * 80}")
        train_loader, val_loader, test_loader = get_pca_data_loaders(
            batch_size=config.BATCH_SIZE,
            n_components=n_features
        )

        # --- ABLATION: ENCODING MODES ---
        for enc_mode in config.ABLATION_ENCODINGS:
            ablation_name = f"enc_{enc_mode}"
            # Unterordner: results/Profilname/Encodingname
            ablation_dir = os.path.join(config.RESULTS_DIR, p_name, ablation_name)
            os.makedirs(ablation_dir, exist_ok=True)
            summary_file_path = os.path.join(ablation_dir, "summary.txt")

            print(f"\n>>> Ablation: {enc_mode}")

            all_histories = []
            test_results = []

            # --- SEED LOOP ---
            for i, seed in enumerate(config.SEEDS):
                print(f"--- [Seed {i + 1}/{len(config.SEEDS)}: {seed}] ---")

                # Reproduzierbarkeit
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Modell initialisieren (Übergabe der Grid-Parameter)
                model = QuantumModel(
                    num_qubits=n_q,
                    num_layers=n_l,
                    init_mode="uniform",
                    encoding_mode=enc_mode
                ).to(config.DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

                # Scheduler mit Patience 8 (etwas aggressiver für Grid Search)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
                )

                criterion = torch.nn.BCEWithLogitsLoss()

                # Training (Inklusive Early Stopping Logik in train_model)
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

                # Evaluation
                test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
                test_results.append({'loss': test_loss, 'acc': test_acc, 'cm': test_cm})

                # Einzelsaat-Logging
                hist_file = os.path.join(ablation_dir, f"seed_{seed}_pred_hist.png")
                plot_prediction_histogram(model, test_loader, hist_file, device=config.DEVICE)

                sample_file = os.path.join(ablation_dir, f"seed_{seed}_sample_outputs.txt")
                log_sample_predictions(model, test_loader, sample_file, device=config.DEVICE, num_batches=5)

                error_plot_path = os.path.join(ablation_dir, f"seed_{seed}_top_errors.png")
                visualize_top_errors(sample_file, raw_test_dataset, num_samples=3, save_path=error_plot_path)

                print(f"Seed {seed} fertig. Acc: {test_acc * 100:.2f}%")

            # --- STATISTISCHE AUSWERTUNG PRO KOMBINATION ---
            f_test_acc = [r['acc'] for r in test_results]
            f_test_loss = [r['loss'] for r in test_results]
            mean_cm = np.mean(np.array([r['cm'] for r in test_results]), axis=0)

            def calc_stats(data):
                return np.mean(data), np.std(data)

            stats_acc = calc_stats(f_test_acc)
            stats_loss = calc_stats(f_test_loss)

            output_str = (
                f"PROFIL: {p_name} | ENCODING: {enc_mode}\n"
                f"{'=' * 40}\n"
                f"TEST ACCURACY: {stats_acc[0] * 100:.2f}% ± {stats_acc[1] * 100:.2f}%\n"
                f"TEST LOSS:     {stats_loss[0]:.4f} ± {stats_loss[1]:.4f}\n"
                f"CONFUSION MATRIX (Mean):\n{mean_cm}\n"
            )

            print(output_str)
            with open(summary_file_path, "w") as f:
                f.write(output_str)

            # Plots für diese Kombination (Encoding + Profil)
            plot_test_accuracy_distribution(f_test_acc, ablation_dir)
            plot_averaged_results(all_histories, ablation_dir)

    print(f"\nGrid Search abgeschlossen. Alle Ergebnisse unter: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()