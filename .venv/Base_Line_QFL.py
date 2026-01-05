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
    plot_average_confusion_matrix,
    plot_test_accuracy_distribution,
    plot_prediction_histogram,
    log_sample_predictions,
    visualize_top_errors,
    analyze_batch_uncertainty,
    plot_master_comparison
)


def main():
    base_dir = config.RESULTS_DIR
    os.makedirs(base_dir, exist_ok=True)

    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    raw_test_dataset = DataClass(split='test', download=True)

    all_ablation_results = []

    for scenario in config.ABLATION_SCENARIOS:
        print(f"\n{'#' * 80}\nSZENARIO: {scenario['name']}\n{'#' * 80}")
        scenario_dir = os.path.join(base_dir, scenario['name'])
        os.makedirs(scenario_dir, exist_ok=True)

        # Daten laden (Features variieren je nach Szenario)
        train_loader, val_loader, test_loader = get_pca_data_loaders(
            batch_size=config.BATCH_SIZE, n_components=scenario['features']
        )

        scenario_histories, scenario_metrics = [], []

        for seed in config.SEEDS:
            print(f"\n--- [Seed: {seed}] ---")
            seed_dir = os.path.join(scenario_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            model = QuantumModelDRU(
                num_qubits=scenario['qubits'],
                num_features=scenario['features'],
                layers_per_block=scenario['layers'],
                encoding_type=scenario['encoding'],
                circuit_type=scenario['type'],
                use_reuploading=scenario['reuploading'],
                trainable_enc=scenario['trainable_enc'],
                measurement=scenario['measurement']
            ).to(config.DEVICE)

            analyze_batch_uncertainty(model, train_loader, config.DEVICE, seed_dir, stage="initial", seed=seed)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

            # --- Dynamischer Loss ---
            if scenario['weighted']:
                # Wir geben der Klasse 0 (Gesund) mehr Gewicht, da sie seltener ist (1:2.8 Ratio)
                # BCELoss kann Gewichte pro Sample verarbeiten
                criterion = torch.nn.BCELoss(reduction='none')
            else:
                criterion = torch.nn.BCELoss()

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

            # Training (Stelle sicher, dass train_model die 'reduction=none' Logik beherrscht)
            history = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                epochs=config.NUM_EPOCHS, seed=seed, device=config.DEVICE,
                is_weighted=scenario['weighted']  # Flag an train_model übergeben
            )
            scenario_histories.append(history)

            analyze_batch_uncertainty(model, train_loader, config.DEVICE, seed_dir, stage="final", seed=seed)

            # Eval (Standard BCELoss für Metriken)
            t_loss, t_acc, t_cm = evaluate_model(model, test_loader, torch.nn.BCELoss(), config.DEVICE)
            scenario_metrics.append({'loss': t_loss, 'acc': t_acc, 'cm': t_cm})

            plot_prediction_histogram(model, test_loader, os.path.join(seed_dir, "hist.png"), device=config.DEVICE)

        plot_averaged_results(scenario_histories, scenario_dir)
        plot_test_accuracy_distribution([m['acc'] for m in scenario_metrics], scenario_dir)

        plot_average_confusion_matrix(scenario_metrics, scenario['name'], scenario_dir)

        print(f"\nSzenario {scenario['name']} beendet.")

        all_ablation_results.append({
            'scenario': scenario['name'],
            'mean_acc': np.mean([m['acc'] for m in scenario_metrics]),
            'std_acc': np.std([m['acc'] for m in scenario_metrics]),
            'history': scenario_histories,
            'metrics': scenario_metrics
        })

        plot_averaged_results(scenario_histories, scenario_dir)
        plot_test_accuracy_distribution([m['acc'] for m in scenario_metrics], scenario_dir)

    print(f"\nERSTELLE MASTER COMPARISON...")
    plot_master_comparison(all_ablation_results, base_dir)


if __name__ == "__main__":
    main()