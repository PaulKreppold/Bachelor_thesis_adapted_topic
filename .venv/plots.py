import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_averaged_results(all_histories, save_dir):
    """Plottet Durchschnitt & Varianz über alle Seeds."""
    epochs = len(all_histories[0]['loss'])
    plt.figure(figsize=(12, 5))

    metrics = [('loss', 'val_loss', 'MSE Loss', 1),
               ('acc', 'val_acc', 'Accuracy', 2)]

    for train_key, val_key, label, idx in metrics:
        plt.subplot(1, 2, idx)
        for key, col, name in [(train_key, 'blue', 'Train'), (val_key, 'orange', 'Val')]:
            data = np.array([h[key] for h in all_histories])
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            plt.plot(range(1, epochs + 1), mean, label=f'{name} {label}', color=col)
            plt.fill_between(range(1, epochs + 1), mean - std, mean + std, color=col, alpha=0.15)

        plt.title(f'Mean {label} (Seeds: {len(all_histories)})')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_training_plot.png'))
    plt.show()


def plot_test_accuracy_distribution(test_accs, results_dir):
    """
    Erstellt einen Boxplot der Test-Accuracy über alle Seeds im Stil von arXiv:2402.09902.
    test_accs: Liste von Floats (0.0 bis 1.0)
    """
    plt.figure(figsize=(6, 8))

    # Daten in Prozent umrechnen
    data = [acc * 100 for acc in test_accs]

    # Style-Einstellungen
    sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"}) # Nur horizontaler Grid

    # Boxplot erstellen
    sns.boxplot(
        y=data,
        width=0.5,
        patch_artist=True, # Ermöglicht das Füllen der Box
        boxprops=dict(facecolor='#a3c1ad', edgecolor='black', linewidth=1.5), # Füllfarbe, schwarzer Rand
        whiskerprops=dict(color='black', linewidth=1.5), # Schwarze Whisker
        capprops=dict(color='black', linewidth=1.5), # Schwarze Kappen
        medianprops=dict(color='black', linewidth=2) # Schwarze, dickere Medianlinie
    )

    # Einzelne Punkte (Seeds) darüber legen für maximale Transparenz
    sns.stripplot(y=data, color="#2a4d34", size=6, jitter=True, edgecolor="black", linewidth=1)

    plt.title("Verteilung der Test-Accuracy über alle Seeds", fontsize=14, pad=20)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)  # Immer 0-100% zeigen für bessere Relation

    # Statistiken als Text einfügen
    median_val = np.median(data)
    plt.text(0.3, median_val, f'Median: {median_val:.2f}%',
             weight='bold', va='center', backgroundcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_accuracy_boxplot.png"), dpi=300)
    plt.close()