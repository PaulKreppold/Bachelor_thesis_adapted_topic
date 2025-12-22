import matplotlib.pyplot as plt

def plot_train_accuracy_and_loss_comparative(train_losses, train_accuracies, seed, client_id, mode_label):
    epochs = range(1, len(train_losses) + 1)
    accuracies_normalized = [acc / 100.0 for acc in train_accuracies]

    if "client_1" in client_id:
        dataset_name = "PneumoniaMNIST"
    elif "client_2" in client_id:
        dataset_name = "RSNA Pneumonia"
    else:
        dataset_name = "MedData"

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'r-', linewidth=2, label='Training Loss')
    plt.plot(epochs, accuracies_normalized, 'b-', linewidth=2, label='Accuracy (0-1)')

    # Titel mit Modus-Angabe (Pure vs. Calibrated)
    plt.title(f'{dataset_name} ({client_id}) - {mode_label}\nLoss & Accuracy Verlauf (Seed {seed})')

    plt.xlabel('Epochen')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()