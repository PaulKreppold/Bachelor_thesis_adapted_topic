import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from scipy.stats import wilcoxon, spearmanr

# Styling
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "figure.figsize": (8, 6)
})


# ============================================================================
# 1. HILFSFUNKTIONEN FÜR LDS
# ============================================================================

def get_minority_info(loader):
    """Extrahiert Minority Class und Ratio."""
    try:
        # Zugriff auf die Labels über das Dataset-Subset (angepasst an data_prep.py)
        labels = loader.dataset.subset.dataset.labels[loader.dataset.subset.indices]
        c = Counter(labels)
        minority_class = 0 if c.get(0, 0) < c.get(1, 0) else 1
        ratio = c.get(minority_class, 0) / len(labels)
        return minority_class, ratio
    except:
        return 1, 0.5


def get_minority_metric_key(loader):
    # Bestimmt, ob 'f1_normal' oder 'f1_krank' die Minority-Klasse ist
    min_class, _ = get_minority_info(loader)
    return "f1_normal" if min_class == 0 else "f1_krank"


def styled_boxplot(ax, data_list, labels, width=0.4):
    bp = ax.boxplot(data_list, patch_artist=True, labels=labels, widths=width)
    colors = ['dodgerblue', 'darkorange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor('white')
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    return bp


def plot_training_convergence(qfl_hists, base_hists, clients, seeds, mode="global"):

    def get_stats(hist_dict, cid):
        # Mean/Std für einen Client über alle Seeds
        series = np.array([hist_dict[s][cid]["train"]["accuracy"] for s in seeds])
        return np.mean(series, axis=0), np.std(series, axis=0)

    if mode == "global":
        plt.figure(figsize=(10, 6))
        # Mittelwert über alle Seeds (axis 0) und alle clients (axis 1)
        all_q = np.mean([[qfl_hists[s][c]["train"]["accuracy"] for c in clients] for s in seeds], axis=(0, 1))
        all_b = np.mean([[base_hists[s][c]["train"]["accuracy"] for c in clients] for s in seeds], axis=(0, 1))

        plt.plot(all_q, label="QFL Global", color="#1E90FF", linewidth=3)
        plt.plot(all_b, label="Baseline Global", color="#FF8C00", linestyle="--", linewidth=2)

        plt.title("Globale Konvergenz (Mittelwert über alle Datensätze & Seeds)")
        plt.xlabel("Epochen (kumuliert)")
        plt.ylabel("Train Accuracy (%)")
        plt.legend()
        plt.show()

    # Pro Silo ein Fenster mit 4 Sub-Clients)
    elif mode == "silo_detailed":
        silos = sorted(list(set([c.split("_sub_")[0] for c in clients])))

        for silo in silos:
            sub_clients = sorted([c for c in clients if c.startswith(silo)], key=lambda x: int(x.split("_sub_")[1]))

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, cid in enumerate(sub_clients):
                ax = axes[idx]
                mu_q, sd_q = get_stats(qfl_hists, cid)
                mu_b, sd_b = get_stats(base_hists, cid)

                ax.plot(mu_q, label="QFL", color="#1E90FF", linewidth=2)
                ax.fill_between(range(len(mu_q)), mu_q - sd_q, mu_q + sd_q, alpha=0.15, color="#1E90FF")

                ax.plot(mu_b, label="Baseline", color="#FF8C00", linestyle="--")
                ax.fill_between(range(len(mu_b)), mu_b - sd_b, mu_b + sd_b, alpha=0.1, color="#FF8C00")

                ax.set_title(f"Sub-Client {cid.split('_sub_')[1]}", fontsize=12)

                ax.set_ylim(0, 105)
                ax.legend(loc='lower right')

            plt.suptitle(f"Konvergenz: Silo {silo.upper()} (Seeds gemittelt)",
                         fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


def plot_minority_f1_boxplots(qfl_res, base_res, clients, loaders, seeds):
    # Erzeugt pro Silo einen Plot für Minority-F1-Scores
    # Konsistenter Split für die Silo-Namen (Index 0)
    silos = sorted(list(set([c.split("_sub_")[0] for c in clients])))

    for silo in silos:
        subs = sorted([c for c in clients if c.startswith(silo)], key=lambda x: int(x.split("_sub_")[1]))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, cid in enumerate(subs):
            ax = axes[idx]

            # Bestimmt ob f1_normal oder f1_krank die Minderheit darstellt
            key = get_minority_metric_key(loaders[cid]['train'])

            # Sammelt F1-Werte über alle Seeds
            vq = [qfl_res[s][cid].get(key, 0) for s in seeds]
            vb = [base_res[s][cid].get(key, 0) for s in seeds]

            # Boxplot-Visualisierung
            styled_boxplot(ax, [vq, vb], ["QFL", "Baseline"])

            # Zusatzinfo zur Ratio im Titel
            _, ratio = get_minority_info(loaders[cid]['train'])
            ax.set_title(f"Client: {cid}\n(Minority Ratio: {ratio:.2%})", fontsize=11)
            ax.set_ylabel(f"F1-Score ({key})")
            ax.set_ylim(0, 1.05)  # F1-Score liegt immer zwischen 0 und 1

        plt.suptitle(f"LDS Breakdown: Silo {silo.upper()} (Minority F1)",
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_f1_vs_minority_ratio(qfl_res, base_res, loaders, seeds):
    # Korrelation zwischen Klassen-Ungleichgewicht und F1-Performance
    ratios, q_means, b_means, q_stds, b_stds = [], [], [], [], []
    for cid in loaders:
        _, ratio = get_minority_info(loaders[cid]['train'])
        key = get_minority_metric_key(loaders[cid]['train'])
        vq = [qfl_res[s][cid].get(key, 0) for s in seeds]
        vb = [base_res[s][cid].get(key, 0) for s in seeds]
        ratios.append(ratio)
        q_means.append(np.mean(vq))
        q_stds.append(np.std(vq))
        b_means.append(np.mean(vb))
        b_stds.append(np.std(vb))
    plt.figure(figsize=(10, 6))
    plt.errorbar(ratios, q_means, yerr=q_stds, fmt='o', label="QFL", color='#1E90FF', capsize=5)
    plt.errorbar(ratios, b_means, yerr=b_stds, fmt='s', label="Baseline", color='#FF8C00', capsize=5, alpha=0.6)
    plt.xlabel("Minority Ratio (Local Imbalance)")
    plt.ylabel("Minority F1-Score (Test)")
    plt.title("LDS Resilience: F1 vs. Imbalance Ratio")
    plt.legend()
    plt.show()


def plot_skew_severity_analysis(qfl_res, base_res, loaders, seeds):
    # Scatter-Plot: Benefit/Gain vs. Skew Severity
    data = []
    for cid, loader in loaders.items():
        _, ratio = get_minority_info(loader['train'])
        skew_severity = abs(0.5 - ratio)
        gains = [qfl_res[s][cid]["f1_macro"] - base_res[s][cid]["f1_macro"] for s in seeds]
        data.append(
            {'Skew': skew_severity, 'Gain': np.mean(gains), 'Dataset': 'MedMNIST' if 'client_1' in cid else 'RSNA'})
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='Skew', y='Gain', scatter_kws={'s': 100}, line_kws={'color': 'red', 'ls': '--'})
    rho, p = spearmanr(df['Skew'], df['Gain'])
    plt.title(f"Gain vs. Skew Severity (rho={rho:.2f}, p={p:.4f})")
    plt.xlabel("Skew Severity (|0.5 - Ratio|)")
    plt.ylabel("Δ F1-Macro (QFL - Baseline)")
    plt.show()


def plot_stability_boxplot(qfl_res, base_res, clients, seeds):
    # Analysiert Streuung der Ø Accuracy über die Seeds
    q_seed_means = [np.mean([qfl_res[s][c]["accuracy"] for c in clients]) for s in seeds]
    b_seed_means = [np.mean([base_res[s][c]["accuracy"] for c in clients]) for s in seeds]
    fig, ax = plt.subplots(figsize=(8, 6))
    styled_boxplot(ax, [q_seed_means, b_seed_means], ['QFL Stability', 'Baseline Stability'])
    ax.set_title("Modell-Stabilität über Seeds (Ø Accuracy pro Seed)", fontweight='bold')
    ax.set_ylabel("Accuracy (%)")
    plt.show()



def plot_global_boxplot(qfl_res, base_res, clients, seeds, central_mean=None):
    all_q = [qfl_res[s][c]["accuracy"] for s in seeds for c in clients]
    all_b = [base_res[s][c]["accuracy"] for s in seeds for c in clients]
    fig, ax = plt.subplots(figsize=(9, 7))
    styled_boxplot(ax, [all_q, all_b], ['QFL (Global)', 'Baseline (Local)'], width=0.5)
    if central_mean:
        ax.axhline(central_mean, color='red', linestyle='--', label=f'Centralized ({central_mean:.1f}%)')
        ax.legend()
    ax.set_title("Globale Performance-Verteilung (Alle Sub-Clients & Seeds)", fontweight='bold')
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 105)
    plt.show()


def plot_confusion_matrix_comparison(qfl_res, base_res, client_id, seeds):
    cm_q = np.sum([qfl_res[s][client_id]["confusion_matrix"] for s in seeds], axis=0)
    cm_b = np.sum([base_res[s][client_id]["confusion_matrix"] for s in seeds], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_q, display_labels=["Norm", "Pneu"]).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title(f"{client_id} - QFL (Sum over Seeds)")
    ConfusionMatrixDisplay(cm_b, display_labels=["Norm", "Pneu"]).plot(ax=axes[1], cmap="Oranges")
    axes[1].set_title(f"{client_id} - Baseline (Sum over Seeds)")
    plt.show()


def compute_scientific_summary(qfl_res, base_res, seeds, clients):

    # Daten flachklopfen für globale Statistik
    all_q_f1 = [qfl_res[s][c]["f1_macro"] for s in seeds for c in clients]
    all_b_f1 = [base_res[s][c]["f1_macro"] for s in seeds for c in clients]

    gains = np.array(all_q_f1) - np.array(all_b_f1)

    # Statistische Signifikanz (Wilcoxon Signed-Rank Test)
    try:
        stat, p_val = wilcoxon(all_q_f1, all_b_f1)
    except ValueError:
        # Fallback falls alle Werte identisch wären
        p_val = 1.0

    # Analyse der stärksten Verbesserung (Max Gain)
    # Berechnung des durchschnittlichen Gain pro Client über alle Seeds
    client_avg_gains = {}
    for cid in clients:
        c_gains = [qfl_res[s][cid]["f1_macro"] - base_res[s][cid]["f1_macro"] for s in seeds]
        client_avg_gains[cid] = np.mean(c_gains)

    best_client = max(client_avg_gains, key=client_avg_gains.get)
    max_gain = client_avg_gains[best_client]

    print("\n" + "█" * 70)
    print(f"{' WISSENSCHAFTLICHE ZUSAMMENFASSUNG (N=' + str(len(all_q_f1)) + ') ':^70}")
    print("█" * 70)

    print(f"{'Metrik':<25} | {'QFL':<12} | {'Baseline':<12} | {'Differenz':<12}")
    print("-" * 70)
    print(f"{'Ø F1-Macro':<25} | {np.mean(all_q_f1):<12.4f} | {np.mean(all_b_f1):<12.4f} | {np.mean(gains):+12.4f}")
    print(
        f"{'Median F1-Macro':<25} | {np.median(all_q_f1):<12.4f} | {np.median(all_b_f1):<12.4f} | {np.median(gains):+12.4f}")
    print(f"{'Std.Abweichung':<25} | {np.std(all_q_f1):<12.4f} | {np.std(all_b_f1):<12.4f} | {'-':<12}")

    print("-" * 70)
    print(f"Statistische Signifikanz (p-Wert): {p_val:.4e}")
    signifikant = "JA" if p_val < 0.05 else "NEIN"
    print(f"Signifikant (p < 0.05): {signifikant}")
    print(f"Win-Rate (QFL > Base): {(gains > 0).mean() * 100:.1f}%")

    print("-" * 70)
    print(f"Stärkster Profit durch QFL: {best_client}")
    print(f"Ø Gain dieses Clients: {max_gain:+.4f} F1-Score Punkte")
    print("█" * 70 + "\n")

    return {
        "avg_gain": np.mean(gains),
        "p_value": p_val,
        "best_client": best_client,
        "win_rate": (gains > 0).mean()
    }


def run_full_evaluation_suite(qfl_res, base_res, qfl_hists, base_hists, clients, loaders, seeds):
    print("\n" + "█" * 60 + f"\n{'STARTE VOLLSTÄNDIGE EVALUATIONS-SUITE':^60}\n" + "█" * 60)
    plot_training_convergence(qfl_hists, base_hists, clients, seeds, mode="global")
    plot_training_convergence(qfl_hists, base_hists, clients, seeds, mode="silo_detailed")
    plot_global_boxplot(qfl_res, base_res, clients, seeds)
    plot_stability_boxplot(qfl_res, base_res, clients, seeds)
    plot_minority_f1_boxplots(qfl_res, base_res, clients, loaders, seeds)
    plot_f1_vs_minority_ratio(qfl_res, base_res, loaders, seeds)
    plot_skew_severity_analysis(qfl_res, base_res, loaders, seeds)
    compute_scientific_summary(qfl_res, base_res, seeds, clients)

    # Beispiel Matrix für den skewed client
    skewed_c = sorted(clients, key=lambda c: get_minority_info(loaders[c]['train'])[1])[0]
    plot_confusion_matrix_comparison(qfl_res, base_res, skewed_c, seeds)
    print("\n" + "█" * 60 + f"\n{'EVALUATION BEENDET':^60}\n" + "█" * 60)