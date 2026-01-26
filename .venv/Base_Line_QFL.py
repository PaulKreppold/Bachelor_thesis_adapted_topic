import torch
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import functools

# Eigene Module
import config
import data_prep
import plots
from Base_Line import QuantumModelNoiseless
from train_and_eval import train_local_model, evaluate_model


def run_seed_ablation(seed):
    """
    Führt für einen einzelnen Seed alle in der Config definierten
    Ablations-Szenarien (Encoding, Measurement, Layer) durch.
    """
    print(f"\n>>> STARTE ABLATION FÜR SEED {seed} (Hardware: {config.DEVICE})")

    # Zentralisierte Daten laden (PneumoniaMNIST)
    # Wir laden 12 Komponenten, das Modell slicet intern je nach Encoding
    train_l, val_l, test_l = data_prep.get_centralized_pneumonia_loaders(
        n_components=config.MAX_FEATURES,
        batch_size=config.BATCH_SIZE,
        seed=seed
    )

    seed_results = []
    seed_histories = {}

    for cfg in config.ABLATION_CONFIGS:
        enc, meas, l = cfg["enc"], cfg["meas"], cfg["L"]

        # Eindeutige Labels für Dateinamen und Legenden
        label = f"L{l}_E-{enc}_M-{meas}"
        full_label = f"S{seed}_{label}"

        # Modell mit spezifischer Layer-Anzahl initialisieren
        torch.manual_seed(seed)
        model = QuantumModelNoiseless(
            n_layers=l,
            encoding_type=enc,
            measurement_type=meas
        ).to(config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

        # 1. Training
        # Gibt state_dict und history {'train_loss', 'train_acc', 'val_loss', 'val_acc'} zurück
        state, history = train_local_model(
            model,
            train_l,
            val_l,
            optimizer,
            config.EPOCHS_BASELINE,
            config.DEVICE,
            client_id=full_label
        )

        # 2. Modell speichern (.pth)
        torch.save(state, os.path.join(config.MODEL_DIR, f"{full_label}.pth"))

        # 3. Evaluierung auf den Testdaten
        test_metrics = evaluate_model(model, test_l, config.DEVICE)

        # 4. Statistiken als JSON speichern
        # Wir bereiten die Metriken für JSON vor (Arrays zu Listen)
        stats = {
            "config": cfg,
            "seed": seed,
            "test_metrics": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in test_metrics.items()},
            "training_history": history
        }

        with open(os.path.join(config.STATS_DIR, f"{full_label}.json"), "w") as f:
            json.dump(stats, f, indent=4)

        # Daten für die Master-CSV und Aggregation sammeln
        seed_results.append({
            "Seed": seed,
            "Layer": l,
            "Encoding": enc,
            "Measurement": meas,
            "Label": label,
            "Testgenauigkeit": test_metrics["Testgenauigkeit"],
            "ROC_AUC": test_metrics["ROC_AUC"],
            "PR_AUC": test_metrics["PR_AUC"]
        })
        seed_histories[label] = history

    return seed_results, seed_histories


def main():
    # Ordnerstruktur initialisieren (über config)
    plots.setup_german_plot_style()

    # Parallelisierung: 2 Seeds gleichzeitig (M1 Pro Auslastung optimieren)
    print(f"Starte Parallel-Processing mit 2 Workern auf {config.DEVICE}...")
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_seed_ablation, config.SEEDS))

    # --- AGGREGATION DER ERGEBNISSE ---
    master_results = []
    all_histories = {}  # Struktur: { 'L2_E-dense_M-all_mean': [hist_seed1, hist_seed2], ... }

    for s_res, s_hist in results:
        master_results.extend(s_res)
        for label, hist in s_hist.items():
            if label not in all_histories:
                all_histories[label] = []
            all_histories[label].append(hist)

    # Master-Ergebnisse als CSV speichern
    df_master = pd.DataFrame(master_results)
    df_master.to_csv(os.path.join(config.BASE_PATH, "Ablation_Master_Results.csv"), index=False)

    # --- PLOTTING ---
    print("\nErzeuge Analyse-Plots...")

    # 1. Trainingskurven getrennt nach Layer (2, 4, 6)
    plots.plot_ablation_training_curves_by_layer(all_histories, config.PLOT_DIR)

    # 2. Globaler Boxplot über alle 18 Konfigurationen
    plots.plot_test_acc_boxplot(df_master, config.PLOT_DIR)

    # 3. Vergleich der Top 3 Gewinner-Konfigurationen
    plots.plot_top_3_configs(all_histories, df_master, config.PLOT_DIR)

    print("\n" + "=" * 60)
    print(f"✔ VORSTUDIE ERFOLGREICH ABGESCHLOSSEN")
    print(f"Speicherort: {config.BASE_PATH}")
    print(f"Modelle: {len(df_master)} .pth Dateien")
    print(f"Stats:   {len(df_master)} .json Dateien")
    print("=" * 60)


if __name__ == "__main__":
    # Wichtig für Multiprocessing auf macOS
    main()