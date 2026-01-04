import torch
import os
import numpy as np
import medmnist
from medmnist import INFO
import config

from data_prep import get_pca_data_loaders
from train_and_eval import train_model, evaluate_model
from Base_Line import QuantumModelDRU

from plots import (
    plot_averaged_results,
    plot_test_accuracy_distribution,
    plot_prediction_histogram,
    log_sample_predictions,
    visualize_top_errors,
    analyze_batch_uncertainty  # <--- Neu importiert
)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Roh-Daten laden
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    raw_test_dataset = DataClass(split='test', download=True)

    # 2. Daten laden (48 PCA-Komponenten laut config)
    print(f"\n[1/3] Lade Daten mit PCA: {config.NUM_FEATURES} Komponenten...")
    train_loader, val_loader, test_loader = get_pca_data_loaders(
        batch_size=config.BATCH_SIZE,
        n_components=config.NUM_FEATURES
    )

    all_histories = []
    test_results = []

    for i, seed in enumerate(config.SEEDS):
        print(f"\n--- [Lauf {i + 1}/{len(config.SEEDS)} | Seed: {seed}] ---")

        # Seed-Ordner sofort erstellen
        seed_dir = os.path.join(config.RESULTS_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # Reproduzierbarkeit
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 3. Modell initialisieren
        model = QuantumModelDRU(
            num_qubits=config.NUM_QUBITS,
            num_features=config.NUM_FEATURES,
            layers_per_block=config.LAYERS_PER_BLOCK
        ).to(config.DEVICE)

        # === A. INITIALE ANALYSE (Untrainierter Zustand) ===
        print("Erstelle initiale Unsicherheits-Analyse...")
        analyze_batch_uncertainty(model, train_loader, config.DEVICE, seed_dir, stage="initial", seed=seed)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7
        )
        criterion = torch.nn.BCELoss()

        # --- 4. TRAINING ---
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

        # === B. FINALE ANALYSE (Nach Training) ===
        print("Erstelle finale Unsicherheits-Analyse...")
        analyze_batch_uncertainty(model, train_loader, config.DEVICE, seed_dir, stage="final", seed=seed)

        # --- 5. EVALUATION ---
        test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
        test_results.append({'loss': test_loss, 'acc': test_acc, 'cm': test_cm})

        # --- 6. WEITERE PLOTS ---
        plot_prediction_histogram(model, test_loader, os.path.join(seed_dir, "pred_hist.png"), device=config.DEVICE)
        sample_log_path = os.path.join(seed_dir, "samples.txt")
        log_sample_predictions(model, test_loader, sample_log_path, device=config.DEVICE)
        visualize_top_errors(sample_log_path, raw_test_dataset, num_samples=3,
                             save_path=os.path.join(seed_dir, "top_errors.png"))

        print(f"Seed {seed} abgeschlossen. Acc: {test_acc * 100:.2f}%")

    # --- 7. GESAMT-AUSWERTUNG ---
    avg_acc = np.mean([r['acc'] for r in test_results])
    std_acc = np.std([r['acc'] for r in test_results])

    plot_averaged_results(all_histories, config.RESULTS_DIR)
    plot_test_accuracy_distribution([r['acc'] for r in test_results], config.RESULTS_DIR)

    # Bericht schreiben
    summary_path = os.path.join(config.RESULTS_DIR, "FINAL_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write(f"ERGEBNISSE VQC DRU\nL_per_Block: {config.LAYERS_PER_BLOCK}\n")
        f.write(f"Mean Acc: {avg_acc * 100:.2f}% +- {std_acc * 100:.2f}%\n")

    print(f"\nTraining beendet. Ergebnisse in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()