import torch
import os
import numpy as np

# Wir laufen beide Encodings nacheinander
ENCODINGS = ["amplitude", "angle_pca"]

for ENCODING in ENCODINGS:

    # Lade passende Config
    if ENCODING == "amplitude":
        import config_amplitude as config
    elif ENCODING == "angle_pca":
        import config_angle_pca as config
    else:
        raise ValueError(f"Unknown encoding: {ENCODING}")

    # Import eigener Module
    from Base_Line import QuantumModel
    from data_prep import get_pneumonia_mnist_loaders, get_pca_angle_loaders
    from train_and_eval import train_model, evaluate_model
    from plots import plot_averaged_results, plot_test_accuracy_distribution

    print(f"\n{'='*80}")
    print(f"START: Training für Encoding → {ENCODING}")
    print(f"{'='*80}\n")

    # Output Ordner
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_file_path = os.path.join(config.RESULTS_DIR, f"final_summary_{ENCODING}.txt")

    # 1️⃣ Daten laden
    if ENCODING == "amplitude":
        train_loader, val_loader, test_loader = get_pneumonia_mnist_loaders(
            batch_size=config.BATCH_SIZE
        )
    elif ENCODING == "angle_pca":
        rsna_root = "/Pfad/zu/RSNA"         # anpassen
        chexpert_root = "/Pfad/zu/CheXpert" # anpassen

        train_loader, val_loader, test_loader = get_federated_pca_loaders(
            rsna_root, chexpert_root,
            n_components=config.NUM_FEATURES,
            batch_size=config.BATCH_SIZE,
            seed=42
        )

    all_histories = []
    test_results = []

    # 2️⃣ Schleife über Seeds
    for i, seed in enumerate(config.SEEDS):
        print(f"--- [SEED {i+1}/{len(config.SEEDS)}: {seed}] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Modell
        model = QuantumModel(
            num_qubits=config.NUM_QUBITS,
            num_layers=config.NUM_LAYERS,
            encoding=ENCODING
        ).to(config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        criterion = torch.nn.BCELoss()

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

        # Test
        test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, config.DEVICE)
        test_results.append({'loss': test_loss, 'acc': test_acc, 'cm': test_cm})

        # Confusion Matrix
        print(f"\n[Seed {seed}] Ergebnisse:")
        print(f"Test-Acc: {test_acc*100:.2f}%")
        print(f"Confusion Matrix (Test):")
        print(f"   TN: {test_cm[0,0]:4d} | FP: {test_cm[0,1]:4d}")
        print(f"   FN: {test_cm[1,0]:4d} | TP: {test_cm[1,1]:4d}")
        print("-"*30+"\n")

    # 3️⃣ Statistik
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
        "Test Acc": calc_stats(f_test_acc)
    }

    # 4️⃣ Ausgabe & Speichern
    output_str = (
        f"{'='*75}\n"
        f"FINALE ERGEBNISSE (Mittelwert ± Std) | Encoding: {ENCODING}\n"
        f"{'='*75}\n"
        f"Train Accuracy:  {stats['Train Acc'][0]*100:.2f}% ± {stats['Train Acc'][1]*100:.2f}%\n"
        f"Val Accuracy:    {stats['Val Acc'][0]*100:.2f}% ± {stats['Val Acc'][1]*100:.2f}%\n"
        f"{'-'*75}\n"
        f"TEST LOSS:       {stats['Test Loss'][0]:.4f} ± {stats['Test Loss'][1]:.4f}\n"
        f"TEST ACCURACY:   {stats['Test Acc'][0]*100:.2f}% ± {stats['Test Acc'][1]*100:.2f}%\n\n"
        f"DURCHSCHNITTLICHE TEST CONFUSION MATRIX:\n"
        f"   TN: {mean_cm[0,0]:6.1f} | FP: {mean_cm[0,1]:6.1f}\n"
        f"   FN: {mean_cm[1,0]:6.1f} | TP: {mean_cm[1,1]:6.1f}\n"
        f"{'='*75}\n"
    )

    print(output_str)
    with open(summary_file_path, "w") as f:
        f.write(output_str)

    # 5️⃣ Plots
    plot_test_accuracy_distribution(f_test_acc, config.RESULTS_DIR)
    plot_averaged_results(all_histories, config.RESULTS_DIR)

    print(f"Run für Encoding '{ENCODING}' abgeschlossen.\n\n")

