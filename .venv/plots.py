import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import csv
import torch
import torch.nn as nn
import pandas as pd


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


def plot_prediction_histogram(model, data_loader, save_path, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.cpu().numpy().ravel())

    plt.figure(figsize=(8, 5))
    # Wir passen die Bins an den Bereich [-1, 1] an
    plt.hist(preds, bins=30, range=(-1.1, 1.1), color="#4c72b0", edgecolor="black", alpha=0.8)

    # Rote Linie bei 0.0, da targets jetzt -1 und 1 sind
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.0)')

    plt.title("Distribution of VQC Predictions (Range [-1, 1])")
    plt.xlabel("VQC Output (PauliZ Expectation Value)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def log_sample_predictions(model, data_loader, save_path, device, num_batches=5):
    """
    Loggt für eine definierte Anzahl Batches die Predictions, Ground Truth
    und den INDIVIDUELLEN MSE-Loss pro Bild.
    """
    model.eval()
    # 'reduction=none' sorgt dafür, dass wir den Loss pro Sample erhalten
    criterion = nn.MSELoss(reduction='none')

    with open(save_path, "w") as f:
        f.write("Sample\tPrediction\tGroundTruth\tIndiv_Loss\n")

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                x, y = x.to(device), y.to(device).float().view(-1, 1)

                out = model(x)
                # Berechnet den quadratischen Fehler für jedes Sample im Batch
                individual_losses = criterion(out, y)

                for i in range(x.shape[0]):
                    # Jetzt loggen wir individual_losses[i] statt batch_loss
                    f.write(
                        f"{batch_idx*len(x)+i}\t"
                        f"{out[i].item():.4f}\t"
                        f"{y[i].item():.0f}\t"
                        f"{individual_losses[i].item():.4f}\n"
                    )


def visualize_top_errors(log_file_path, raw_dataset, num_samples=3, save_path=None):
    df = pd.read_csv(log_file_path, sep='\t')

    # Wir sortieren nach dem höchsten Indiv_Loss
    top_errors = df.sort_values(by='Indiv_Loss', ascending=False).head(num_samples)

    plt.figure(figsize=(15, 6))

    for i, (idx, row) in enumerate(top_errors.iterrows()):
        sample_idx = int(row['Sample'])
        pred = row['Prediction']
        # Beachte: Wenn im Log GT 0/1 steht, müssen wir das für den Vergleich
        # zu -1/1 mappen, falls das Modell auf -1/1 trainiert wurde.
        gt = row['GroundTruth']
        loss = row['Indiv_Loss']

        img, _ = raw_dataset[sample_idx]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(np.array(img), cmap='gray')

        # Entscheidungsschwelle 0:
        # Wenn Pred > 0 und GT = 1 (Pneumonia) -> Richtig
        # Wenn Pred <= 0 und GT = 0 (Normal) -> Richtig
        is_correct = (pred > 0 and gt == 1) or (pred <= 0 and gt == 0)

        color = 'green' if is_correct else 'red'
        title = f"Idx: {sample_idx}\nPred: {pred:.4f} | GT: {int(gt)}\nLoss: {loss:.4f}"
        plt.title(title, color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle("Analyse der größten Abweichungen (Labels mappen auf -1 / +1)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()