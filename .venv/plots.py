import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_averaged(seed_histories, config_str, save_path=None):
    """
    Erstellt einen kombinierten Plot für Loss und Accuracy über mehrere Seeds.

    Args:
        seed_histories: Liste von Dictionaries (5 Stück, eines pro Seed).
        config_str: String mit den Hyperparametern (z.B. "BCE | AllQubits | L:4 | LR:0.001").
        save_path: Der Pfad inklusive Dateiname (file_id), unter dem gespeichert wird.
    """
    # Anzahl der Epochen bestimmen (nimmt die Länge des ersten Seeds)
    epochs = range(1, len(seed_histories[0]['train_loss']) + 1)
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']

    # Daten für die Berechnung in Numpy-Arrays umwandeln (Shape: Seeds x Epochen)
    data = {m: np.array([h[m] for h in seed_histories]) for m in metrics}

    # Mittelwert und Standardabweichung berechnen
    means = {m: np.mean(data[m], axis=0) for m in metrics}
    stds = {m: np.std(data[m], axis=0) for m in metrics}

    # Plot-Layout initialisieren
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ÜBERGEORDNETER TITEL MIT HYPERPARAMETERN
    fig.suptitle(f"Ablationsstudie: {config_str}\n(Mittelwert ± Std über {len(seed_histories)} Seeds)",
                 fontsize=14, fontweight='bold', y=0.98)

    # --- 1. LOSS PLOT (Links) ---
    # Train Loss
    ax1.plot(epochs, means['train_loss'], color='#1f77b4', label='Train Loss', linewidth=2)
    ax1.fill_between(epochs, means['train_loss'] - stds['train_loss'],
                     means['train_loss'] + stds['train_loss'], color='#1f77b4', alpha=0.2)

    # Val Loss
    ax1.plot(epochs, means['val_loss'], color='#ff7f0e', label='Val Loss', linewidth=2, linestyle='--')
    ax1.fill_between(epochs, means['val_loss'] - stds['val_loss'],
                     means['val_loss'] + stds['val_loss'], color='#ff7f0e', alpha=0.2)

    ax1.set_title('Loss Verlauf', fontsize=12)
    ax1.set_xlabel('Epochen')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 2. ACCURACY PLOT (Rechts) ---
    # Train Acc
    ax2.plot(epochs, means['train_acc'], color='#2ca02c', label='Train Acc', linewidth=2)
    ax2.fill_between(epochs, means['train_acc'] - stds['train_acc'],
                     means['train_acc'] + stds['train_acc'], color='#2ca02c', alpha=0.2)

    # Val Acc
    ax2.plot(epochs, means['val_acc'], color='#d62728', label='Val Acc', linewidth=2, linestyle='--')
    ax2.fill_between(epochs, means['val_acc'] - stds['val_acc'],
                     means['val_acc'] + stds['val_acc'], color='#d62728', alpha=0.2)

    ax2.set_title('Accuracy Verlauf', fontsize=12)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('Genauigkeit')
    ax2.set_ylim([0.45, 1.02])  # Da Binärklassifizierung (Zufall = 0.5)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Layout optimieren, damit Titel nicht abgeschnitten wird
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Speichern des Bildes (der Pfad kommt aus der main.py und enthält die Hyperparameter)
    if save_path:
        plt.savefig(save_path, dpi=200)

    # WICHTIG: Speicher freigeben (verhindert RAM-Probleme bei 48 Plots)
    plt.close(fig)