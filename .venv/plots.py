import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch


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


def plot_ablation_training_comparison(all_scenarios_data, save_dir):
    """
    Vergleicht alle Ablation-Szenarien in zwei Plots (Train-Loss und Train-Acc).
    Jede Linie ist der Durchschnitt über die Seeds des jeweiligen Szenarios.
    """
    plt.figure(figsize=(16, 6))

    # 1. Plot für Train Loss
    plt.subplot(1, 2, 1)
    for data in all_scenarios_data:
        # data['history'] ist eine Liste von Dicts (eines pro Seed)
        all_losses = [h['loss'] for h in data['history']]
        # Mittelwert über Seeds berechnen
        mean_loss = np.mean(all_losses, axis=0)
        plt.plot(range(1, len(mean_loss) + 1), mean_loss, label=data['scenario'])

    plt.title('Train Loss Vergleich (Mittelwert über Seeds)')
    plt.xlabel('Epochen')
    plt.ylabel('BCE Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)

    # 2. Plot für Train Accuracy
    plt.subplot(1, 2, 2)
    for data in all_scenarios_data:
        all_accs = [h['acc'] for h in data['history']]
        mean_acc = np.mean(all_accs, axis=0)
        plt.plot(range(1, len(mean_acc) + 1), mean_acc, label=data['scenario'])

    plt.title('Train Accuracy Vergleich (Mittelwert über Seeds)')
    plt.xlabel('Epochen')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_training_comparison.png"), dpi=300)
    plt.close()


def plot_master_comparison(all_results, save_dir):
    """Balkendiagramm der Test-Accuracy mit korrigierter 62.5% Baseline."""
    df = pd.DataFrame(all_results)
    df['mean_acc_pct'] = df['mean_acc'] * 100
    df['std_acc_pct'] = df['std_acc'] * 100
    df = df.sort_values(by='mean_acc_pct', ascending=True)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    bars = plt.barh(df['scenario'], df['mean_acc_pct'],
                    xerr=df['std_acc_pct'],
                    color='skyblue', edgecolor='navy', alpha=0.8, capsize=5)

    # KORREKTUR: Test-Set Baseline
    plt.axvline(x=62.5, color='red', linestyle='--', linewidth=2,
                label='Majority Class Test Baseline (62.5%)')

    plt.title('VQC Ablationsstudie: Vergleich der Test-Accuracy', fontsize=16, pad=20)
    plt.xlabel('Test Accuracy (%)', fontsize=12)
    plt.xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2f}%', va='center', fontsize=10, fontweight='bold')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "master_ablation_comparison.png"), dpi=300)
    plt.close()

