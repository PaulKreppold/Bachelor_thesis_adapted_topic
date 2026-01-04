import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn


def plot_averaged_results(all_histories, save_dir):
    """Plottet Durchschnitt & Varianz über alle Seeds für ein einzelnes Experiment."""
    max_epochs = max(len(h['loss']) for h in all_histories)
    plt.figure(figsize=(12, 5))
    metrics = [('loss', 'val_loss', 'BCE Loss', 1),
               ('acc', 'val_acc', 'Accuracy', 2)]

    for train_key, val_key, label, idx in metrics:
        plt.subplot(1, 2, idx)
        for key, col, name in [(train_key, 'blue', 'Train'), (val_key, 'orange', 'Val')]:
            padded_data = []
            for h in all_histories:
                series = list(h[key])
                if len(series) < max_epochs:
                    last_val = series[-1]
                    series.extend([last_val] * (max_epochs - len(series)))
                padded_data.append(series)

            data = np.array(padded_data)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            epochs_range = range(1, max_epochs + 1)
            plt.plot(epochs_range, mean, label=f'{name} {label}', color=col)
            plt.fill_between(epochs_range, mean - std, mean + std, color=col, alpha=0.15)

        plt.title(f'Mean {label}')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def plot_master_comparison(all_exp_data, save_dir):
    """
    Erstellt den 'Master-Plot': Vergleicht alle 4 Experimente in einer Grafik.
    all_exp_data: Liste von Dicts [{'name': str, 'histories': list}, ...]
    """
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # Farben und Styles für die 4 Varianten
    styles = {
        "Enc-angle_Scale-False": ("blue", "--"),
        "Enc-angle_Scale-True": ("blue", "-"),
        "Enc-iqp_Scale-False": ("red", "--"),
        "Enc-iqp_Scale-True": ("red", "-")
    }

    metrics = [('val_loss', 'Validation Loss', 1), ('val_acc', 'Validation Accuracy', 2)]

    for key, label, idx in metrics:
        plt.subplot(1, 2, idx)
        for exp in all_exp_data:
            name = exp['name']
            histories = exp['histories']
            color, ls = styles.get(name, ("black", "-"))

            # Mittelwert über Seeds berechnen (mit Padding)
            max_len = max(len(h[key]) for h in histories)
            padded = []
            for h in histories:
                series = list(h[key])
                series += [series[-1]] * (max_len - len(series))
                padded.append(series)

            mean_series = np.mean(padded, axis=0)
            plt.plot(range(1, max_len + 1), mean_series, label=name, color=color, linestyle=ls, linewidth=2)

        plt.title(f"Comparison: {label}", fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'MASTER_COMPARISON_PLOT.png'), dpi=300)
    plt.close()


def plot_test_accuracy_distribution(test_accs, results_dir):
    """Boxplot der Test-Accuracy über alle Seeds."""
    plt.figure(figsize=(6, 8))
    data = [acc * 100 for acc in test_accs]
    sns.set_theme(style="whitegrid")
    sns.boxplot(y=data, width=0.5, color='#a3c1ad')
    sns.stripplot(y=data, color="#2a4d34", size=6, jitter=True)
    plt.title("Test-Accuracy Distribution", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_accuracy_boxplot.png"), dpi=300)
    plt.close()


def plot_prediction_histogram(model, data_loader, save_path, device):
    """Verteilung der VQC-Rohausgaben."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in data_loader:
            out = model(x.to(device))
            preds.extend(out.cpu().numpy().ravel())
    plt.figure(figsize=(8, 5))
    plt.hist(preds, bins=30, color="#4c72b0", edgecolor="black", alpha=0.8)
    plt.axvline(x=0.0, color='red', linestyle='--', label='Threshold')
    plt.title("Distribution of Predictions")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def log_sample_predictions(model, data_loader, save_path, device, num_batches=5):
    """Speichert Samples für die Error-Visualisierung."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    with open(save_path, "w") as f:
        f.write("Sample\tPrediction\tGroundTruth\tIndiv_Loss\n")
        with torch.no_grad():
            for b_idx, (x, y) in enumerate(data_loader):
                if b_idx >= num_batches: break
                y_float = y.to(device).float().view(-1, 1)
                out = model(x.to(device))
                losses = criterion(out, y_float)
                for i in range(x.shape[0]):
                    f.write(f"{b_idx * len(x) + i}\t{out[i].item():.4f}\t{y[i].item():.0f}\t{losses[i].item():.4f}\n")


def visualize_top_errors(log_file_path, raw_dataset, num_samples=3, save_path=None):
    """Zeigt Bilder mit höchstem Loss."""
    if not os.path.exists(log_file_path): return
    df = pd.read_csv(log_file_path, sep='\t')
    top_errors = df.sort_values(by='Indiv_Loss', ascending=False).head(num_samples)
    plt.figure(figsize=(15, 6))
    for i, (idx, row) in enumerate(top_errors.iterrows()):
        img, _ = raw_dataset[int(row['Sample'])]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(np.array(img), cmap='gray')
        pred, gt = row['Prediction'], row['GroundTruth']
        color = 'green' if (pred > 0 and gt == 1) or (pred <= 0 and gt == 0) else 'red'
        plt.title(f"Pred: {pred:.2f} | GT: {int(gt)}\nLoss: {row['Indiv_Loss']:.3f}", color=color)
        plt.axis('off')
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

