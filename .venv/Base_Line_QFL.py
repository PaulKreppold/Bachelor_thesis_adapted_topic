import torch
import torch.nn as nn
import os
import json
import time  # Neu für die Zeitmessung
import config
from data_prep import get_mnist_binary_loaders
from Base_Line import QuantumModel
from train_and_eval import train_model
from plots import plot_metrics_averaged
from summarize_results import generate_summary


def pad_history(history, max_epochs):
    """Füllt gekürzte Histories nach Early Stopping auf."""
    new_history = {}
    for key in history.keys():
        values = list(history[key])
        last_val = values[-1]
        while len(values) < max_epochs:
            values.append(last_val)
        new_history[key] = values
    return new_history


def main():
    # Ordner erstellen
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Daten laden
    train_loader, val_loader, _ = get_mnist_binary_loaders(batch_size=config.BATCH_SIZE)

    # --- Fortschritts-Setup ---
    total_experiments = (
                len(config.STUDY_LOSS_MODES) * len(config.STUDY_MEASURE_MODES) * len(config.STUDY_LAYERS) * len(
            config.STUDY_LRS))

    current_idx = 0
    start_time_total = time.time()

    print(f"{'=' * 70}")
    print(f"STARTE QUANTUM VQC ABLATIONSSTUDIE")
    print(f"Gesamtanzahl Experimente: {total_experiments} (je {len(config.SEEDS)} Seeds)")
    print(f"Speicherpfad: {config.RESULTS_DIR}")
    print(f"{'=' * 70}\n")

    for loss_mode in config.STUDY_LOSS_MODES:
        use_bce = (loss_mode == "BCE")
        criterion = nn.BCELoss() if use_bce else nn.MSELoss()

        for measure_all in config.STUDY_MEASURE_MODES:
            m_label = "AllQubits" if measure_all else "SingleQubit"

            for layers in config.STUDY_LAYERS:
                for lr in config.STUDY_LRS:
                    current_idx += 1
                    exp_start_time = time.time()

                    config_desc = f"{loss_mode} | {m_label} | Layers: {layers} | LR: {lr}"
                    file_id = f"{loss_mode}_{m_label}_L{layers}_LR{lr}"

                    print(f"[{current_idx}/{total_experiments}] Experiment: {file_id}")
                    print(f"{'-' * 70}")

                    seed_histories = []

                    for seed in config.SEEDS:
                        print(f"  -> Seed {seed}:")
                        torch.manual_seed(seed)

                        model = QuantumModel(config.NUM_QUBITS, layers, measure_all, use_bce).to(config.DEVICE)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        # Hier wird nun kompakt (alle 10 Epochen) geprintet
                        history = train_model(
                            model, train_loader, val_loader, optimizer,
                            criterion, config.EPOCHS_PER_EXP, use_bce, config.EARLY_STOPPING_PATIENCE
                        )

                        seed_histories.append(pad_history(history, config.EPOCHS_PER_EXP))

                    # Nach 5 Seeds: Plotten und Speichern
                    plot_path = os.path.join(config.RESULTS_DIR, f"{file_id}.png")
                    plot_metrics_averaged(seed_histories, config_desc, save_path=plot_path)

                    with open(os.path.join(config.RESULTS_DIR, f"{file_id}_data.json"), "w") as f:
                        json.dump(seed_histories, f)

                    # --- Zeit-Analyse nach jedem Experiment ---
                    exp_duration = (time.time() - exp_start_time) / 60
                    elapsed_total_hrs = (time.time() - start_time_total) / 3600
                    avg_time_per_exp = elapsed_total_hrs / current_idx
                    remaining_exps = total_experiments - current_idx
                    est_remaining_hrs = remaining_exps * avg_time_per_exp

                    print(f"\n  [✓] Fertig. Dauer: {exp_duration:.2f} Min")
                    print(
                        f"  [i] Fortschritt: {current_idx / total_experiments * 100:.1f}% | Est. Restzeit: {est_remaining_hrs:.2f} Std\n")

    # Finale Zusammenfassung (CSV)
    print(f"\n{'=' * 70}")
    print("ALLE EXPERIMENTE BEENDET. GENERIERE ZUSAMMENFASSUNG...")
    generate_summary(results_dir=config.RESULTS_DIR, output_file=os.path.join(config.RESULTS_DIR, "summary.csv"))

    total_duration_hrs = (time.time() - start_time_total) / 3600
    print(f"Gesamtdauer: {total_duration_hrs:.2f} Stunden")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()