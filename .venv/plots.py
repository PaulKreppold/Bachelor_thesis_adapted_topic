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


def plot_average_confusion_matrix(scenario_metrics, scenario_name, save_dir):
    """
    Berechnet die durchschnittliche Confusion Matrix über alle Seeds eines Szenarios
    und plottet diese als Heatmap.
    """
    # 1. Alle CMs extrahieren und summieren
    all_cms = [m['cm'] for m in scenario_metrics]
    avg_cm = np.mean(all_cms, axis=0)

    # 2. Normalisierung (Prozentual pro Ground Truth Zeile)
    # Zeigt: Wie viel Prozent der Gesunden wurden als Gesund erkannt?
    cm_perc = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))

    # Labels für die Matrix
    labels = ['Gesund (0)', 'Pneumonie (1)']

    # Annotations erstellen (Kombination aus Absolutwert und Prozent)
    annot = [
        [f"{avg_cm[0, 0]:.1f}\n({cm_perc[0, 0]:.1%})", f"{avg_cm[0, 1]:.1f}\n({cm_perc[0, 1]:.1%})"],
        [f"{avg_cm[1, 0]:.1f}\n({cm_perc[1, 0]:.1%})", f"{avg_cm[1, 1]:.1f}\n({cm_perc[1, 1]:.1%})"]
    ]

    sns.heatmap(cm_perc, annot=annot, fmt="", cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=True)

    plt.title(f"Average Confusion Matrix\nSzenario: {scenario_name}")
    plt.ylabel('Ground Truth')
    plt.xlabel('VQC Vorhersage')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Average Confusion Matrix für {scenario_name} gespeichert.")


def plot_master_comparison(all_results, save_dir):
    """
    Erstellt einen Balkendiagramm-Vergleich aller Ablation-Szenarien.
    all_results: Liste von Dicts mit 'scenario', 'mean_acc', 'std_acc'
    """
    # Daten in DataFrame umwandeln
    df = pd.DataFrame(all_results)

    # Sortieren für bessere Optik (optional)
    df = df.sort_values(by='mean_acc', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Balkendiagramm mit Fehlerbalken
    bars = plt.bar(df['scenario'], df['mean_acc'] * 100,
                   yerr=df['std_acc'] * 100,
                   capsize=10, color='skyblue', edgecolor='navy', alpha=0.8)

    # Werte über die Balken schreiben
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1,
                 f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Experimentelles Szenario')
    plt.title('Ablationsstudie: Vergleich der VQC-Architekturen (PneumoniaMNIST)')

    # 74% Referenzlinie (Majority Class Bias)
    plt.axhline(y=74.0, color='red', linestyle='--', label='Majority Class Baseline (74%)')

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "master_ablation_comparison.png"))
    plt.close()

    # Tabelle als CSV speichern für die Thesis-Anhänge
    df[['scenario', 'mean_acc', 'std_acc']].to_csv(
        os.path.join(save_dir, "ablation_results_table.csv"), index=False
    )
    print(f"Master-Vergleich gespeichert in {save_dir}")


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
    criterion = nn.BCELoss(reduction='none')
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
        color = 'green' if (pred >= 0.5 and gt == 1) or (pred < 0.5 and gt == 0) else 'red'
        plt.title(f"Pred: {pred:.2f} | GT: {int(gt)}\nLoss: {row['Indiv_Loss']:.3f}", color=color)
        plt.axis('off')
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_batch_uncertainty(model, loader, device, save_dir, stage="initial", seed=None):
    """
    Analysiert den ersten Batch auf Unsicherheit und speichert Tabelle + Plot inkl. Seed-Info.
    """
    model.eval()
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets_float = targets.view(-1, 1).float().to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss_fn = nn.BCELoss(reduction='none')
        individual_losses = loss_fn(outputs, targets_float)

    outputs_np = outputs.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    losses_np = individual_losses.cpu().numpy().flatten()

    # DataFrame erstellen und SEED hinzufügen
    df = pd.DataFrame({
        'Seed': [seed] * len(inputs),  # Neue Spalte für den Seed
        'ID': range(len(inputs)),
        'Pred': outputs_np,
        'GT': targets_np,
        'Loss': losses_np,
        'Dist_0.5': np.abs(outputs_np - 0.5)
    }).sort_values(by='Dist_0.5')

    # Dateiname enthält nun auch den Seed zur eindeutigen Identifizierung
    csv_filename = f"uncertainty_{stage}_seed_{seed}.csv"
    df.to_csv(os.path.join(save_dir, csv_filename), index=False)

    # Plotting mit Seed im Titel
    plt.figure(figsize=(10, 5))
    sns.stripplot(x=outputs_np, y=targets_np.astype(str), hue=targets_np,
                  palette={0: 'blue', 1: 'red'}, jitter=0.1, alpha=0.7, orient='h', order=['0', '1'])

    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.axvspan(0.4, 0.6, color='yellow', alpha=0.1)

    plt.title(f"Unsicherheits-Analyse ({stage.upper()}) | Seed: {seed}")
    plt.xlabel("Vorhersage Wahrscheinlichkeit (Pneumonie)")
    plt.ylabel("Ground Truth Label")
    plt.xlim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plot_filename = f"uncertainty_plot_{stage}_seed_{seed}.png"
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close()