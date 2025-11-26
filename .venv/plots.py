import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc, average_precision_score
import seaborn as sns
import pandas as pd
import torch


def plot_qfl_vs_baseline_accuracy(qfl_accuracies, baseline_accuracies, clients):
    """
    Plottet die Trainingsgenauigkeit von QFL vs. Baseline,
    wobei maximal 4 Clients (in einem 2x2 Grid) pro Seite angezeigt werden.
    """

    clients_sorted = sorted(clients)  # Clients sortieren für eine klare Reihenfolge
    clients_per_page = 4

    # Clients in Batches von 4 verarbeiten
    for page_num, i in enumerate(range(0, len(clients_sorted), clients_per_page)):

        clients_batch = clients_sorted[i:i + clients_per_page]
        num_clients_on_page = len(clients_batch)

        # Grid-Größe für diese Seite festlegen (max. 2x2)
        if num_clients_on_page <= 2:
            nrows, ncols = 1, 2
            figsize = (16, 8)
        else:
            nrows, ncols = 2, 2
            figsize = (16, 14)

        # Abbildung erstellen
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, client_id in enumerate(clients_batch):
            ax = axes[idx]

            # --- QFL Datenverarbeitung ---
            qfl_curves = []
            for seed in qfl_accuracies:
                qfl_curves.append(qfl_accuracies[seed][client_id])

            min_len_qfl = min(len(c) for c in qfl_curves)
            qfl_curves_trimmed = np.array([c[:min_len_qfl] for c in qfl_curves])

            mean_qfl = np.mean(qfl_curves_trimmed, axis=0)
            std_qfl = np.std(qfl_curves_trimmed, axis=0)
            epochs_qfl = np.arange(len(mean_qfl))

            # Plot QFL
            ax.plot(epochs_qfl, mean_qfl, label="QFL Training", color="dodgerblue")
            ax.fill_between(epochs_qfl, mean_qfl - std_qfl, mean_qfl + std_qfl,
                            alpha=0.2, color="dodgerblue")

            # --- Baseline Datenverarbeitung ---
            baseline_curves = []
            for seed in baseline_accuracies:
                baseline_curves.append(baseline_accuracies[seed][client_id])

            min_len_baseline = min(len(c) for c in baseline_curves)
            baseline_curves_trimmed = np.array([c[:min_len_baseline] for c in baseline_curves])

            mean_baseline = np.mean(baseline_curves_trimmed, axis=0)
            std_baseline = np.std(baseline_curves_trimmed, axis=0)
            epochs_baseline = np.arange(len(mean_baseline))

            # Plot Baseline
            ax.plot(epochs_baseline, mean_baseline, label="Baseline Training", color="darkorange")
            ax.fill_between(epochs_baseline, mean_baseline - std_baseline, mean_baseline + std_baseline,
                            alpha=0.2, color="darkorange")

            # --- Achsen und Beschriftung ---
            ax.set_title(f"Vergleich für Client {client_id}")
            ax.set_xlabel("Epochen")
            ax.set_ylabel("Trainings-Genauigkeit (%)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            ax.set_ylim(50, 95)  # Konstantes Limit für besseren Vergleich

        # Entferne ungenutzte Subplots auf der letzten Seite (falls weniger als 4 Clients)
        for j in range(num_clients_on_page, len(axes)):
            fig.delaxes(axes[j])

        # Titel und Layout für die gesamte Seite
        fig.suptitle(f"QFL vs. Baseline: Trainingsgenauigkeit (Seite {page_num + 1})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()  # Zeige die aktuelle Seite


def plot_per_client_boxplots(qfl_client_results, baseline_client_results, clients):
    """
    Plottet automatisch alle Subclients in 4er-Gruppen pro Seite.
    Der Base-Client wird automatisch aus dem Subclient-Namen extrahiert.
    """
    # Sortiere Clients nach Base-Client und Subclient-Nummer, falls noch nicht sortiert
    clients_sorted = sorted(clients, key=lambda x: (x.split('_sub')[0], int(x.split('_sub')[1])))

    # Gehe in 4er-Batches
    for i in range(0, len(clients_sorted), 4):
        subclients_batch = clients_sorted[i:i+4]
        base_client_name = subclients_batch[0].split('_sub')[0]

        num_clients = len(subclients_batch)
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), squeeze=False)
        axes = axes.flatten()
        box_linewidth = 1.2

        # Für dynamische Skalierung
        all_qfl_page_values = []
        all_baseline_page_values = []

        for client_id in subclients_batch:
            qfl_values = [qfl_client_results[seed][client_id] for seed in qfl_client_results]
            baseline_values = [baseline_client_results[seed][client_id] for seed in baseline_client_results]
            all_qfl_page_values.extend(qfl_values)
            all_baseline_page_values.extend(baseline_values)

        if not all_qfl_page_values or not all_baseline_page_values:
            min_val, max_val = 60, 85
        else:
            min_val = min(min(all_qfl_page_values), min(all_baseline_page_values)) - 1
            max_val = max(max(all_qfl_page_values), max(all_baseline_page_values)) + 1
            if max_val - min_val < 5:
                max_val += 2
                min_val -= 2

        # Plotten der einzelnen Subclients
        for idx, client_id in enumerate(subclients_batch):
            ax = axes[idx]
            qfl_values = [qfl_client_results[seed][client_id] for seed in qfl_client_results]
            baseline_values = [baseline_client_results[seed][client_id] for seed in baseline_client_results]

            bp = ax.boxplot([qfl_values, baseline_values], patch_artist=True, notch=False)
            for box in bp['boxes']:
                box.set(facecolor='white', edgecolor='black', linewidth=box_linewidth)
            for whisker in bp['whiskers']:
                whisker.set(color='black', linewidth=box_linewidth)
            for cap in bp['caps']:
                cap.set(color='black', linewidth=box_linewidth)
            for median in bp['medians']:
                median.set(color='black', linewidth=box_linewidth + 0.5)
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)

            ax.set_xticklabels(['QFL (globales Modell)', 'Baseline (lokal)'])
            ax.set_ylabel('Test-Genauigkeit (%)')
            ax.set_title(f'sub client {client_id}: QFL vs. Baseline')
            ax.yaxis.grid(True, linestyle='--', alpha=0.25)
            ax.set_ylim(min_val, max_val)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        for j in range(num_clients, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'base client {base_client_name}: Vergleich pro sub client (QFL vs. Baseline)',
                     fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



def plot_stability_boxplot(qfl_results_across_seeds, baseline_results_across_seeds, central_acc_per_seed):
    # 1. Datenvorbereitung: Sammeln der Seed-Durchschnitte (N=5 pro Liste)
    qfl_acc_list = list(qfl_results_across_seeds.values())
    baseline_acc_list = list(baseline_results_across_seeds.values())
    central_acc_list = list(central_acc_per_seed.values())

    data_to_plot = [qfl_acc_list, baseline_acc_list, central_acc_list]

    # 2. Erstellung und Styling des Boxplots
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False)

    box_linewidth = 1.2
    for box in bp['boxes']:
        box.set(facecolor='white', edgecolor='black', linewidth=box_linewidth)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=box_linewidth)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=box_linewidth)
    for median in bp['medians']:
        median.set(color='black', linewidth=box_linewidth + 0.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)

    # 3. Beschriftung anpassen
    ax.set_xticklabels(['QFL', 'Lokale Baseline', 'Zentrale Baseline'])
    ax.set_ylabel('Durchschnittliche Test-Genauigkeit (%)')
    ax.set_title('Vergleich der Algorithmus-Stabilität (Seed-Variabilität)')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_rsna_auc_stability(qfl_rsna_auc, baseline_rsna_auc, central_rsna_auc):
    # 1. Datenvorbereitung
    # Wir nehmen an, die Metriken sind Dictionaries {seed: value}
    qfl_auc_list = list(qfl_rsna_auc.values())
    baseline_auc_list = list(baseline_rsna_auc.values())
    central_auc_list = list(central_rsna_auc.values())

    data_to_plot = [qfl_auc_list, baseline_auc_list, central_auc_list]

    # 2. Erstellung und Styling des Boxplots
    fig, ax = plt.subplots(figsize=(8, 6))

    # patch_artist=True ist entscheidend für die Füllfarbe
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False)

    # Styling-Code aus plot_global_boxplot übertragen:
    box_linewidth = 1.2

    # Definiere die Farben für Median und Boxen, um sie schwarz/weiß zu machen
    for box in bp['boxes']:
        box.set(facecolor='white', edgecolor='black', linewidth=box_linewidth)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=box_linewidth)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=box_linewidth)
    for median in bp['medians']:
        median.set(color='black', linewidth=box_linewidth + 0.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)

    # 3. Beschriftung anpassen
    ax.set_xticklabels(['QFL', 'Lokale Baseline', 'Zentrale Baseline'])
    ax.set_ylabel('auc score auf externem RSNA Testset')
    ax.set_title('Vergleich der Algorithmus-Stabilität (AUC Score RSNA)')

    # Hinzufügen der horizontalen Gridlines (wie im ersten Plot)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)

    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.show()


# --- ANGEPASSTE FUNKTION plot_global_boxplot (20 vs 20 + Linie) ---
def plot_global_boxplot(qfl_client_results, baseline_client_results, clients, central_overall_mean):
    all_qfl = [qfl_client_results[seed][client_id]
               for seed in qfl_client_results for client_id in clients]
    all_baseline = [baseline_client_results[seed][client_id]
                    for seed in baseline_client_results for client_id in clients]

    data_to_plot = [all_qfl, all_baseline]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False)

    # Styling (Code wie zuvor)
    box_linewidth = 1.2
    for box in bp['boxes']:
        box.set(facecolor='white', edgecolor='black', linewidth=box_linewidth)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=box_linewidth)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=box_linewidth)
    for median in bp['medians']:
        median.set(color='black', linewidth=box_linewidth + 0.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)

    # NEU: Hinzufügen der zentralen Obergrenze als horizontale Linie
    ax.axhline(central_overall_mean, color='red', linestyle='--', linewidth=2, label='Zentrale Obergrenze')

    ax.set_xticklabels(['Quantum Federated Learning', 'Baseline (Lokal)'])
    ax.set_ylabel('Finale Test-Genauigkeit (%)')
    ax.set_title('Aggregierter Vergleich: QFL vs. Baseline (über alle Clients & Seeds)')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def identify_skewed_clients(client_train_loaders, threshold=0.5):
    """
    Identifiziert Clients mit stark unbalancierten Klassenverteilungen.
    Unterstützt RSNADataset, ImagePathDataset, ImageFolder und verschachtelte Subsets.
    - threshold: minimale Fraction der Minoritätsklasse, die noch als 'ausgeglichen' gilt.
    """
    def get_base_dataset_and_indices(ds):
        """Entpackt rekursiv alle Subsets, bis das Basis-Dataset erreicht ist."""
        indices = None
        while isinstance(ds, torch.utils.data.Subset):
            indices = ds.indices if indices is None else [ds.indices[i] for i in indices]
            ds = ds.dataset
        return ds, indices

    def extract_labels(ds, indices=None):
        """Extrahiert Labels aus einem Dataset, egal ob RSNADataset, ImagePathDataset oder ImageFolder."""
        if hasattr(ds, "labels"):
            labels = np.array(ds.labels)
        elif hasattr(ds, "targets"):
            labels = np.array(ds.targets)
        else:
            # Fallback: versuche label durch direkten Zugriff zu holen
            try:
                labels = np.array([ds[i][1] for i in range(len(ds))])
            except Exception:
                return None
        if indices is not None:
            labels = labels[indices]
        return labels.astype(int)

    skewed_clients = []

    for cid, loader in client_train_loaders.items():
        ds = loader.dataset
        base_ds, indices = get_base_dataset_and_indices(ds)
        labels = extract_labels(base_ds, indices)

        if labels is None or len(labels) == 0:
            print(f"[WARNUNG] {cid}: Labels konnten nicht extrahiert werden – übersprungen.")
            continue

        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            print(f"[INFO] {cid}: nur eine Klasse vorhanden – als stark verzerrt markiert.")
            skewed_clients.append(cid)
            continue
        if len(unique) > 2:
            print(f"[WARNUNG] {cid}: {len(unique)} Klassen gefunden (nicht binär). Überspringe...")
            continue

        frac_minority = min(counts) / np.sum(counts)
        print(f"[INFO] {cid}: class_counts={dict(zip(unique, counts))} → minority_frac={frac_minority:.3f} (threshold={threshold})")

        if frac_minority < threshold:
            skewed_clients.append(cid)

    return skewed_clients



def plot_confusion_matrix_skewed(qfl_metrics, baseline_metrics, skewed_clients):
    """
    Zeigt Confusion-Matrizen (QFL vs. Baseline) für alle skewed clients nebeneinander.
    """
    for cid in skewed_clients:
        if cid not in qfl_metrics or cid not in baseline_metrics:
            print(f"Client {cid} fehlt in den Ergebnissen – wird übersprungen.")
            continue

        qfl_cm = qfl_metrics[cid]["confusion_matrix"]
        base_cm = baseline_metrics[cid]["confusion_matrix"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Client {cid} – Confusion Matrix (QFL vs. Baseline)", fontsize=14)

        ConfusionMatrixDisplay(qfl_cm, display_labels=["Normal", "Pneumonie"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("QFL Clientmodell")

        ConfusionMatrixDisplay(base_cm, display_labels=["Normal", "Pneumonie"]).plot(ax=axes[1], cmap="Oranges", colorbar=False)
        axes[1].set_title("Baseline Modell")

        plt.tight_layout()
        plt.show()


def plot_roc_pr_skewed(qfl_metrics, baseline_metrics, skewed_clients):
    """
    Zeichnet ROC- und Precision/Recall-Kurven für alle skewed Clients.
    Vergleicht QFL-Clientmodell vs. Baseline-Client.
    """
    for cid in skewed_clients:
        if cid not in qfl_metrics or cid not in baseline_metrics:
            print(f"Client {cid} fehlt in den Ergebnissen – wird übersprungen.")
            continue

        # Daten vorbereiten
        y_true_qfl = np.array(qfl_metrics[cid]["labels"]).flatten()
        y_probs_qfl = np.array(qfl_metrics[cid]["probs"]).flatten()
        y_true_base = np.array(baseline_metrics[cid]["labels"]).flatten()
        y_probs_base = np.array(baseline_metrics[cid]["probs"]).flatten()

        # ROC
        fpr_qfl, tpr_qfl, _ = roc_curve(y_true_qfl, y_probs_qfl)
        fpr_base, tpr_base, _ = roc_curve(y_true_base, y_probs_base)
        auc_qfl = auc(fpr_qfl, tpr_qfl)
        auc_base = auc(fpr_base, tpr_base)

        # PR
        prec_qfl, rec_qfl, _ = precision_recall_curve(y_true_qfl, y_probs_qfl)
        prec_base, rec_base, _ = precision_recall_curve(y_true_base, y_probs_base)

        # Plotten
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Client {cid} – ROC & Precision/Recall", fontsize=14)

        # ROC
        axes[0].plot(fpr_qfl, tpr_qfl, label=f"QFL (AUC={auc_qfl:.3f})", color="blue")
        axes[0].plot(fpr_base, tpr_base, label=f"Baseline (AUC={auc_base:.3f})", color="orange")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_title("ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend()

        # PR
        axes[1].plot(rec_qfl, prec_qfl, label="QFL", color="blue")
        axes[1].plot(rec_base, prec_base, label="Baseline", color="orange")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


def plot_auc_heatmap_skewed(qfl_metrics, baseline_metrics, skewed_clients):
    """
    Zeigt Heatmap der AUC-Werte (QFL vs. Baseline) für alle skewed Clients.
    """
    data = []
    for cid in skewed_clients:
        if cid not in qfl_metrics or cid not in baseline_metrics:
            continue
        data.append({
            "Client": cid,
            "QFL AUC": qfl_metrics[cid]["auc"],
            "Baseline AUC": baseline_metrics[cid]["auc"]
        })

    if not data:
        print("Keine gültigen Clients für Heatmap gefunden.")
        return

    df = pd.DataFrame(data).set_index("Client")

    plt.figure(figsize=(6, len(df) * 0.6 + 2))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title("AUC-Vergleich (QFL vs. Baseline) – Skewed Clients")
    plt.xlabel("Modell")
    plt.ylabel("Client")
    plt.tight_layout()
    plt.show()


def plot_auc_pr_heatmap_skewed(qfl_metrics, baseline_metrics, skewed_clients):
    """
    Zeigt eine Heatmap der gespeicherten AUC-PR-Werte (QFL vs. Baseline) für alle
    Clients mit Label-Skew. Nutzt die direkt gespeicherten 'auc_pr' Metriken.
    """
    data = []

    # 1. Daten aus den gespeicherten Metriken sammeln
    for cid in skewed_clients:
        # Prüfung, ob Client-Metriken für den aktuellen Seed vorhanden sind
        if cid not in qfl_metrics or cid not in baseline_metrics:
            continue

        # Abruf der gespeicherten AUC-PR Werte
        # (Fehlerbehandlung: Prüfen, ob der Schlüssel 'auc_pr' existiert)
        auc_pr_qfl = qfl_metrics[cid].get("auc_pr", np.nan)
        auc_pr_base = baseline_metrics[cid].get("auc_pr", np.nan)

        data.append({
            "Client": cid,
            "QFL AUC-PR": auc_pr_qfl,
            "Baseline AUC-PR": auc_pr_base
        })

    if not data:
        print("Keine gültigen Clients für AUC-PR Heatmap gefunden.")
        return

    # 2. DataFrame erstellen und Heatmap plotten
    df = pd.DataFrame(data).set_index("Client")

    plt.figure(figsize=(6, len(df) * 0.6 + 2))
    # 'viridis' ist eine gute Farbskala für kontinuierliche Daten
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".3f", vmin=0.5)
    plt.title("AUC-PR Vergleich (QFL vs. Baseline) – Skewed Clients")
    plt.xlabel("Modell")
    plt.ylabel("Client")
    plt.tight_layout()
    plt.show()

def compute_bias_summary(qfl_client_metrics, baseline_client_metrics, qfl_rsna_metrics, central_global_metrics, seeds):

    bias_results = {}

    print("\n\n BIAS & FAIRNESS ANALYSE\n")

    for seed in seeds:
        qfl_clients = qfl_client_metrics[seed]
        base_clients = baseline_client_metrics[seed]
        rsna_qfl = qfl_rsna_metrics[seed]
        central_rsna = central_global_metrics[seed]

        # Client-Heterogenität
        f1_qfl_values = np.array([m["f1"] for m in qfl_clients.values()])
        f1_base_values = np.array([m["f1"] for m in base_clients.values()])
        auc_qfl_values = np.array([m["auc"] for m in qfl_clients.values()])

        f1_var_qfl = np.std(f1_qfl_values)
        f1_var_base = np.std(f1_base_values)

        # Delta Global vs. Local (pro Client)
        delta_f1 = np.mean([qfl_clients[cid]["f1"] - base_clients[cid]["f1"] for cid in qfl_clients])
        delta_auc = np.mean([qfl_clients[cid]["auc"] - base_clients[cid]["auc"] for cid in qfl_clients])

        # RSNA Generalization Gap
        f1_global_rsna = rsna_qfl["global_model"]["f1"]
        f1_central_rsna = central_rsna["f1"]
        avg_f1_clients = np.mean(f1_qfl_values)
        generalization_gap = avg_f1_clients - f1_global_rsna

        # Precision/Recall Imbalance
        imbalance_qfl = np.mean([abs(m["precision"] - m["recall"]) for m in qfl_clients.values()])
        imbalance_base = np.mean([abs(m["precision"] - m["recall"]) for m in base_clients.values()])

        bias_results[seed] = {
            "F1_Var_QFL": f1_var_qfl,
            "F1_Var_Baseline": f1_var_base,
            "ΔF1_QFL-Base": delta_f1,
            "ΔAUC_QFL-Base": delta_auc,
            "RSNA_Gap_F1": generalization_gap,
            "Precision/Recall_Imbalance_QFL": imbalance_qfl,
            "Precision/Recall_Imbalance_Base": imbalance_base,
            "Central_RSNA_F1": f1_central_rsna
        }

        print(f"\n Seed {seed}")
        print(f"  • Client-Heterogenität (σ F1): QFL={f1_var_qfl:.4f} | Baseline={f1_var_base:.4f}")
        print(f"  • Δ(QFL - Baseline): F1={delta_f1:+.3f} | AUC={delta_auc:+.3f}")
        print(f"  • RSNA Generalization Gap (avg client F1 → RSNA): {generalization_gap:+.3f}")
        print(f"  • Precision/Recall Imbalance: QFL={imbalance_qfl:.3f} | Baseline={imbalance_base:.3f}")
        print(f"  • Central RSNA F1: {f1_central_rsna:.3f}")

    print("\n Aggregierte Bias-Statistik über alle Seeds:")
    df_bias = pd.DataFrame(bias_results).T
    print(df_bias.round(4).to_string())

    print("\n Interpretation:")
    print(" Niedrigere F1-Varianz → homogenere Performance (weniger client bias).")
    print(" Positives ΔF1 → QFL übertrifft lokale Baselines (kollektive Verbesserung).")
    print(" Kleinerer RSNA-Gap → bessere Generalisierung auf externe Domäne.")
    print(" Kleinere Precision/Recall-Differenz → besser kalibriertes Modell (weniger Label-Bias).")

    return df_bias

