import torch
import os
import numpy as np
import config  # Importiert SEEDS, BATCH_SIZE, NUM_QUBITS, NUM_LAYERS, LR, etc.
from data_prep import get_pneumonia_mnist_loaders
from Base_Line import QuantumModel
from train_and_eval import train_model, evaluate_model
from plots import plot_averaged_results, plot_test_accuracy_distribution

def main():
    # 0. Vorbereitung
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_file_path = os.path.join(config.RESULTS_DIR, "final_summary.txt")

    # 1. Daten laden (Gibt train, val und test loader zurück)
    train_loader, val_loader, test_loader = get_pneumonia_mnist_loaders(batch_size=config.BATCH_SIZE)

    all_histories = []
    test_results = []

    print(f"{'=' * 75}")
    print(f"STARTE QUANTUM VQC RUN")
    print(f"Datensatz: PneumoniaMNIST")
    print(f"Device: {config.DEVICE} | Epochs: {config.NUM_EPOCHS} | Seeds: {len(config.SEEDS)}")
    print(f"Konfiguration: {config.NUM_QUBITS} Qubits, {config.NUM_LAYERS} Layers, LR: {config.LR}")
    print(f"{'=' * 75}\n")

    # 2. Schleife über die konfigurierten Seeds
    for i, seed in enumerate(config.SEEDS):
        print(f"--- [SEED {i + 1}/{len(config.SEEDS)}: {seed}] ---")

        # Reproduzierbarkeit
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Modell, Optimizer & Criterion
        model = QuantumModel(
            num_qubits=config.NUM_QUBITS,
            num_layers=config.NUM_LAYERS
        ).to(config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Training
        history, final_val_cm = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config.NUM_EPOCHS,
            seed=seed,
            device=config.DEVICE
        )
        all_histories.append(history)

        # Test-Evaluation (Unseen Data)
        test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
        test_results.append({'loss': test_loss, 'acc': test_acc, 'cm': test_cm})

        # --- AUSGABE DER CONFUSION MATRIX FÜR DIESEN SEED ---
        print(f"\n[Seed {seed}] Ergebnisse:")
        print(f"Test-Acc: {test_acc * 100:.2f}%")
        print(f"Confusion Matrix (Test):")
        print(f"   TN: {test_cm[0, 0]:4d} | FP: {test_cm[0, 1]:4d}")
        print(f"   FN: {test_cm[1, 0]:4d} | TP: {test_cm[1, 1]:4d}")
        print("-" * 30 + "\n")

    # 3. Statistische Auswertung berechnen
    f_train_acc = [h['acc'][-1] for h in all_histories]
    f_val_acc = [h['val_acc'][-1] for h in all_histories]
    f_test_acc = [r['acc'] for r in test_results]
    f_test_loss = [r['loss'] for r in test_results]

    # Durchschnittliche CM über alle Seeds berechnen
    all_test_cms = np.array([r['cm'] for r in test_results])
    mean_cm = np.mean(all_test_cms, axis=0)

    def calc_stats(data):
        return np.mean(data), np.std(data)

    stats = {
        "Train Acc": calc_stats(f_train_acc),
        "Val Acc":   calc_stats(f_val_acc),
        "Test Loss": calc_stats(f_test_loss),
        "Test Acc":  calc_stats(f_test_acc),
    }

    # 4. Ergebnisausgabe & Speichern
    output_str = (
        f"{'=' * 75}\n"
        f"FINALE ERGEBNISSE (Mittelwert ± Standardabweichung)\n"
        f"{'=' * 75}\n"
        f"Train Accuracy:  {stats['Train Acc'][0] * 100:.2f}% ± {stats['Train Acc'][1] * 100:.2f}%\n"
        f"Val Accuracy:    {stats['Val Acc'][0] * 100:.2f}% ± {stats['Val Acc'][1] * 100:.2f}%\n"
        f"{'-' * 75}\n"
        f"TEST LOSS:       {stats['Test Loss'][0]:.4f} ± {stats['Test Loss'][1]:.4f}\n"
        f"TEST ACCURACY:   {stats['Test Acc'][0] * 100:.2f}% ± {stats['Test Acc'][1] * 100:.2f}%\n\n"
        f"DURCHSCHNITTLICHE TEST CONFUSION MATRIX:\n"
        f"   TN: {mean_cm[0, 0]:6.1f} | FP: {mean_cm[0, 1]:6.1f}\n"
        f"   FN: {mean_cm[1, 0]:6.1f} | TP: {mean_cm[1, 1]:6.1f}\n"
        f"{'=' * 75}\n"
    )

    print(output_str)

    with open(summary_file_path, "w") as f:
        f.write(output_str)

    # 5. Visualisierung
    print("Erstelle Plots...")
    plot_test_accuracy_distribution(f_test_acc, config.RESULTS_DIR)
    plot_averaged_results(all_histories, config.RESULTS_DIR)

    print(f"Alles erledigt. Ergebnisse gespeichert unter: {config.RESULTS_DIR}")

# Der fehlende Aufruf:
if __name__ == "__main__":
    main()