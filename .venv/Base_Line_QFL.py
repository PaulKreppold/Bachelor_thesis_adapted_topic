import torch
import os
import numpy as np
import config
from data_prep import get_pca_data_loaders
from train_and_eval import train_model, evaluate_model
from Base_Line import UniversalVQC
import plots


def main():
    # Liste für die Master-Auswertung
    all_ablation_results = []

    for scenario in config.ABLATION_SCENARIOS:
        print(f"\n{'#' * 60}\nSZENARIO: {scenario['name']}\n{'#' * 60}")
        scenario_dir = os.path.join(config.RESULTS_DIR, scenario['name'])
        os.makedirs(scenario_dir, exist_ok=True)

        # PCA Dimension berechnen
        n_feats = scenario['qubits'] * (2 if scenario['encoding'] == 'dense' else 1)
        train_loader, val_loader, test_loader = get_pca_data_loaders(
            batch_size=config.BATCH_SIZE, n_components=n_feats
        )

        scenario_histories = []
        scenario_metrics = []

        for seed in config.SEEDS:
            seed_dir = os.path.join(scenario_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = UniversalVQC(scenario).to(config.DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=scenario['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

            # Training
            history = train_model(model, train_loader, val_loader, optimizer, scheduler,
                                  config.NUM_EPOCHS, seed, config.DEVICE, scenario)
            scenario_histories.append(history)

            # Evaluation
            crit = torch.nn.CrossEntropyLoss() if scenario['measurement'] == 'softmax' else torch.nn.BCELoss()
            loss, acc, cm = evaluate_model(model, test_loader, crit, config.DEVICE, scenario)
            scenario_metrics.append({'loss': loss, 'acc': acc, 'cm': cm})

            plots.plot_prediction_histogram(model, test_loader, os.path.join(seed_dir, "hist.png"), config.DEVICE)

        # Lokale Auswertung für dieses Szenario
        plots.plot_averaged_results(scenario_histories, scenario_dir)
        plots.plot_average_confusion_matrix(scenario_metrics, scenario['name'], scenario_dir)

        # WICHTIG: Daten für den Master-Vergleich sammeln
        all_ablation_results.append({
            'scenario': scenario['name'],
            'mean_acc': np.mean([m['acc'] for m in scenario_metrics]),
            'std_acc': np.std([m['acc'] for m in scenario_metrics]),
            'history': scenario_histories  # Für den Trainingsverlauf-Vergleich
        })

    # --- FINALE MASTER-AUSWERTUNG ---
    print("\nERSTELLE GESAMT-VERGLEICHE...")
    plots.plot_ablation_training_comparison(all_ablation_results, config.RESULTS_DIR)
    plots.plot_master_comparison(all_ablation_results, config.RESULTS_DIR)
    print(f"\nAlle Analysen abgeschlossen. Ergebnisse in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()