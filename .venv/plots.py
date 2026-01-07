import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch


def plot_averaged_results(all_histories, save_dir):
    """
    Plottet den Durchschnitt und die Standardabweichung von Loss und Accuracy
    über alle Seeds hinweg für das finale Modell.
    """
    max_epochs = max(len(h['loss']) for h in all_histories)
    plt.figure(figsize=(12, 5))

    # Metriken: (Train-Key, Val-Key, Label, Subplot-Index)
    metrics = [('loss', 'val_loss', 'BCE Loss', 1),
               ('acc', 'val_acc', 'Accuracy', 2)]

    for train_key, val_key, label, idx in metrics:
        plt.subplot(1, 2, idx)
        for key, col, name in [(train_key, 'blue', 'Train'), (val_key, 'orange', 'Val')]:
            padded_data = []
            for h in all_histories:
                series = list(h[key])
                # Padding, falls ein Seed durch Early Stopping kürzer lief
                if len(series) < max_epochs:
                    last_val = series[-1]
                    series.extend([last_val] * (max_epochs - len(series)))
                padded_data.append(series)

            data = np.array(padded_data)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            epochs_range = range(1, max_epochs + 1)

            plt.plot(epochs_range, mean, label=f'{name} {label}', color=col, linewidth=2)
            plt.fill_between(epochs_range, mean - std, mean + std, color=col, alpha=0.15)

        plt.title(f'Mean {label} (über alle Seeds)')
        plt.xlabel('Epochen')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def plot_average_confusion_matrix(scenario_metrics, scenario_name, save_dir):
    """Visualisiert die Performance bei der Erkennung beider Klassen (Normal/Pneumonie)."""
    all_cms = [m['cm'] for m in scenario_metrics]
    avg_cm = np.mean(all_cms, axis=0)
    cm_perc = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    labels = ['Gesund (0)', 'Pneumonie (1)']
    annot = [
        [f"{avg_cm[0, 0]:.1f}\n({cm_perc[0, 0]:.1%})", f"{avg_cm[0, 1]:.1f}\n({cm_perc[0, 1]:.1%})"],
        [f"{avg_cm[1, 0]:.1f}\n({cm_perc[1, 0]:.1%})", f"{avg_cm[1, 1]:.1f}\n({cm_perc[1, 1]:.1%})"]
    ]

    sns.heatmap(cm_perc, annot=annot, fmt="", cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=True)

    plt.title(f"Durchschnittliche Confusion Matrix\nModell: {scenario_name}")
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('VQC Vorhersage')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_confusion_matrix.png'), dpi=300)
    plt.close()


def plot_test_accuracy_distribution(test_accs, results_dir):
    """Zeigt die statistische Varianz der Test-Accuracy über die Seeds (Boxplot)."""
    plt.figure(figsize=(6, 8))
    data = [acc * 100 for acc in test_accs]
    sns.set_theme(style="whitegrid")

    # Kombination aus Boxplot und Stripplot für maximale Transparenz
    sns.boxplot(y=data, width=0.5, color='#a3c1ad')
    sns.stripplot(y=data, color="#2a4d34", size=8, jitter=True)

    plt.title("Verteilung der Test-Accuracy", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_accuracy_boxplot.png"), dpi=300)
    plt.close()


def plot_prediction_histogram(model, data_loader, save_path, device):
    """Zeigt die Konfidenz-Verteilung der Wahrscheinlichkeiten (0.0 bis 1.0)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in data_loader:
            out = model(x.to(device))
            preds.extend(out.cpu().numpy().ravel())

    plt.figure(figsize=(8, 5))
    plt.hist(preds, bins=30, color="#4c72b0", edgecolor="black", alpha=0.8)
    # WICHTIG: Threshold bei 0.5 für Wahrscheinlichkeiten
    plt.axvline(x=0.5, color='red', linestyle='--', label='Entscheidungsgrenze (0.5)')

    plt.title("Verteilung der Vorhersage-Wahrscheinlichkeiten")
    plt.xlabel("Wahrscheinlichkeit (0.0=Gesund, 1.0=Pneumonie)")
    plt.ylabel("Häufigkeit")
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_final_statistics(metrics, save_dir):
    """Speichert die finalen statistischen Kennzahlen in einer Textdatei."""
    accs = [m['acc'] * 100 for m in metrics]
    losses = [m['loss'] for m in metrics]

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_loss = np.mean(losses)

    stats_path = os.path.join(save_dir, "final_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("=== Finale Statistik des Super-Modells ===\n")
        f.write(f"Test Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)\n")
        f.write(f"Durchschnittlicher Test Loss: {mean_loss:.4f}\n")
        f.write(f"Anzahl Seeds: {len(metrics)}\n")

    print(f"\n[Statistik] Finale Kennzahlen gespeichert: {stats_path}")
    print(f"Ergebnis: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")


def save_raw_results(all_histories, all_metrics, save_dir):
    """Speichert Rohdaten als .npy, damit der globale Plotter darauf zugreifen kann."""
    # Durchschnittliche Historie berechnen
    max_epochs = max(len(h['loss']) for h in all_histories)
    avg_history = {}
    for key in ['loss', 'acc', 'val_loss', 'val_acc']:
        data = []
        for h in all_histories:
            series = list(h[key])
            if len(series) < max_epochs:
                series.extend([series[-1]] * (max_epochs - len(series)))
            data.append(series)
        avg_history[key] = np.mean(data, axis=0)

    # Speichern der Durchschnitts-Historie
    np.save(os.path.join(save_dir, 'avg_history.npy'), avg_history)

    # Speichern aller Test-Accuracies der Seeds (für Boxplot)
    test_accs = [m['acc'] for m in all_metrics]
    np.save(os.path.join(save_dir, 'test_accs.npy'), test_accs)

    # Speichern der gemittelten Confusion Matrix
    avg_cm = np.mean([m['cm'] for m in all_metrics], axis=0)
    np.save(os.path.join(save_dir, 'avg_cm.npy'), avg_cm)


def plot_global_comparison(results_root, layer_configs, init_methods):
    """Erstellt den großen Vergleich über alle Experimente."""
    all_box_data = []
    plt.figure(figsize=(15, 10))
    fig_line, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))

    for layers in layer_configs:
        for init in init_methods:
            exp_name = f"Layers_{layers}_Init_{init}"
            path = os.path.join(results_root, exp_name)

            if not os.path.exists(os.path.join(path, 'avg_history.npy')):
                continue

            # 1. Daten für Boxplot sammeln
            accs = np.load(os.path.join(path, 'test_accs.npy'))
            for a in accs:
                all_box_data.append({'Layers': layers, 'Init': init, 'Test Acc': a * 100})

            # 2. Daten für Liniendiagramme
            hist = np.load(os.path.join(path, 'avg_history.npy'), allow_pickle=True).item()
            label = f"L{layers}-{init}"
            ax_loss.plot(hist['loss'], label=label, alpha=0.7)
            ax_acc.plot(hist['acc'], label=label, alpha=0.7)

    # Styling Liniendiagramme
    ax_loss.set_title("Globaler Train Loss Vergleich")
    ax_loss.set_xlabel("Epoche");
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.legend(fontsize='x-small', ncol=2)
    ax_acc.set_title("Globaler Train Accuracy Vergleich")
    ax_acc.set_xlabel("Epoche");
    ax_acc.set_ylabel("Accuracy")
    fig_line.tight_layout()
    fig_line.savefig(os.path.join(results_root, "global_training_comparison.png"))

    # Styling Boxplot
    plt.figure(figsize=(12, 7))
    df = pd.DataFrame(all_box_data)
    sns.boxplot(x='Layers', y='Test Acc', hue='Init', data=df)
    plt.title("Test Accuracy Vergleich: Layer-Tiefe vs. Initialisierung")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_root, "global_test_boxplot.png"))
    plt.close('all')


def plot_global_cm_grid(results_root, layer_configs, init_methods):
    """Erstellt ein Grid aus Confusion Matrices für alle Varianten."""
    n_rows = len(layer_configs)
    n_cols = len(init_methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # Falls nur eine Zeile/Spalte existiert, axes array-kompatibel machen
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    for i, layers in enumerate(layer_configs):
        for j, init in enumerate(init_methods):
            exp_name = f"Layers_{layers}_Init_{init}"
            path = os.path.join(results_root, exp_name, 'avg_cm.npy')

            ax = axes[i, j]
            if os.path.exists(path):
                cm = np.load(path)
                # Normalisieren für Farbgebung
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(f"L{layers} - {init}", fontsize=10)
            else:
                ax.axis('off')
            ax.set_xticks([]);
            ax.set_yticks([])

    plt.suptitle("Confusion Matrix Vergleich (normalisiert)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_root, "global_cm_grid.png"))
    plt.close()

