import matplotlib.pyplot as plt


def plot_train_accuracy_and_loss(train_losses, train_accuracies, seed):
    """
    Plottet Trainings-Loss und Trainings-Accuracy (skaliert auf 0-1)
    als durchgezogene Linien (ohne Punkte).
    """
    epochs = range(1, len(train_losses) + 1)

    # WICHTIG: Accuracy von 0-100% in 0.0-1.0 umrechnen
    accuracies_normalized = [acc / 100.0 for acc in train_accuracies]

    plt.figure(figsize=(10, 6))

    # --- ÄNDERUNGEN HIER ---
    # 'r-' erzeugt eine rote Linie ohne Marker (Punkte)
    # linewidth=2 macht die Linie etwas dicker (paper-style)
    plt.plot(epochs, train_losses, 'r-', linewidth=2, label='Training Loss')

    # 'b-' erzeugt eine blaue Linie ohne Marker
    plt.plot(epochs, accuracies_normalized, 'b-', linewidth=2, label='Accuracy (Skaliert 0-1)')
    # -----------------------

    plt.title(f'Baseline Training: Loss & Accuracy Verlauf (Seed {seed})')
    plt.xlabel('Epochen')

    # Y-Achsen Beschriftung entfernt, wie gewünscht
    # plt.ylabel('Wert (0.0 - 1.0)')

    # Gitter und Legende
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center right')

    # Y-Achse fest auf 0 bis 1.05
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()