import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

# Import aus data_prep
from data_prep import get_labels_from_dataset

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
    "boxplot.boxprops.linewidth": 1.2,
    "boxplot.whiskerprops.linewidth": 1.2,
    "boxplot.capprops.linewidth": 1.2,
    "boxplot.medianprops.linewidth": 1.5,
    "boxplot.flierprops.markersize": 4,
    "figure.figsize": (8, 6)
})


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def get_labels_local(dataset):
    """Alias für get_labels_from_dataset für Rückwärtskompatibilität"""
    return get_labels_from_dataset(dataset)


def get_minority_class(loader):
    try:
        labels = get_labels_local(loader.dataset)
        if len(labels) == 0: return 1
        c = Counter(labels)
        return 0 if c.get(0, 0) < c.get(1, 0) else 1
    except:
        return 1


def get_minority_ratio(loader):
    labels = get_labels_local(loader.dataset)
    if len(labels) == 0: return 0.0
    c = Counter(labels)
    minority = min(c.values())
    return minority / len(labels)


def identify_skewed_clients(client_train_loaders, threshold=0.3):
    return [cid for cid, loader in client_train_loaders.items() if get_minority_ratio(loader) < threshold]


def styled_boxplot(ax, data_list, labels, width=0.25):
    """Konsistenter Paper-Style Boxplot"""
    bp = ax.boxplot(data_list,
                    patch_artist=True,
                    labels=labels,
                    notch=False,
                    widths=width)

    for box in bp['boxes']:
        box.set_facecolor('white')
        box.set_edgecolor('black')
        box.set_linewidth(1.2)

    for median in bp['medians']:
        median.set_color('orange')
        median.set_linewidth(2.0)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)

    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5, markersize=3)
        flier.set_markeredgecolor('black')

    return bp


# ============================================================================
# BESTEHENDE PLOT-FUNKTIONEN
# ============================================================================

def plot_qfl_vs_baseline_accuracy(qfl_accuracies, baseline_accuracies, clients):
    """Accuracy über Epochen (Mittel ± STD) pro Client."""
    clients_sorted = sorted(clients)
    per_page = 4

    for i in range(0, len(clients_sorted), per_page):
        batch = clients_sorted[i:i + per_page]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, cid in enumerate(batch):
            ax = axes[idx]

            arr_q = np.array([qfl_accuracies[s][cid] for s in qfl_accuracies])
            min_len = min(len(c) for c in arr_q)
            arr_q = arr_q[:, :min_len]
            mu_q, sd_q = arr_q.mean(0), arr_q.std(0)

            arr_b = np.array([baseline_accuracies[s][cid] for s in baseline_accuracies])
            min_len_b = min(len(c) for c in arr_b)
            arr_b = arr_b[:, :min_len_b]
            mu_b, sd_b = arr_b.mean(0), arr_b.std(0)

            ax.plot(mu_q, label="QFL", color="dodgerblue")
            ax.fill_between(range(len(mu_q)), mu_q - sd_q, mu_q + sd_q,
                            alpha=0.25, color="dodgerblue")

            ax.plot(mu_b, label="Baseline", color="darkorange")
            ax.fill_between(range(len(mu_b)), mu_b - sd_b, mu_b + sd_b,
                            alpha=0.25, color="darkorange")

            ax.set_title(cid)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        for k in range(len(batch), 4):
            fig.delaxes(axes[k])

        plt.tight_layout()
        plt.show()


def plot_per_client_boxplots(qfl_client_results, baseline_client_results, clients):
    """Erstellt separate Plots für jeden Base-Client mit seinen 4 Sub-Clients."""
    base_clients_dict = {}
    for cid in clients:
        try:
            base = cid.rsplit("_sub", 1)[0]
            if base not in base_clients_dict:
                base_clients_dict[base] = []
            base_clients_dict[base].append(cid)
        except:
            print(f"Warning: Konnte {cid} nicht parsen, überspringe.")
            continue

    sorted_bases = sorted(base_clients_dict.keys())
    for base in base_clients_dict:
        try:
            base_clients_dict[base] = sorted(
                base_clients_dict[base],
                key=lambda x: int(x.rsplit("_sub", 1)[1])
            )
        except:
            base_clients_dict[base] = sorted(base_clients_dict[base])

    for base in sorted_bases:
        sub_clients = base_clients_dict[base]
        num_subs = len(sub_clients)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, cid in enumerate(sub_clients):
            ax = axes[idx]

            qfl_values = [qfl_client_results[s][cid]["accuracy"] for s in qfl_client_results]
            baseline_values = [baseline_client_results[s][cid]["accuracy"] for s in baseline_client_results]

            styled_boxplot(ax, [qfl_values, baseline_values], ["QFL", "Baseline"], width=0.4)

            sub_num = cid.rsplit("_sub", 1)[1]
            ax.set_ylabel('Test Accuracy', fontsize=11)
            ax.set_title(f'Sub-Client {sub_num}', fontweight='bold', fontsize=12)
            ax.set_ylim([0, 1])
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax.xaxis.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for i in range(num_subs, 4):
            fig.delaxes(axes[i])

        fig.suptitle(
            f'{base.upper()} - QFL vs. Baseline Performance',
            fontsize=15, fontweight='bold', y=0.995
        )

        plt.tight_layout()
        plt.show()
        print(f"✓ Plot für {base} erstellt ({num_subs} Sub-Clients)")


def plot_global_boxplot(qfl_client_results, baseline_client_results, clients, central_mean=None):
    all_q = [qfl_client_results[s][c]["accuracy"] for s in qfl_client_results for c in clients]
    all_b = [baseline_client_results[s][c]["accuracy"] for s in baseline_client_results for c in clients]

    fig, ax = plt.subplots(figsize=(8, 6))
    styled_boxplot(ax, [all_q, all_b], ['QFL (Global)', 'Baseline (Local)'], width=0.4)

    if central_mean:
        ax.axhline(central_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Centralized ({central_mean:.1f}%)')
        ax.legend()

    ax.set_title("Global Performance Distribution")
    ax.set_ylabel("Test Accuracy")
    plt.show()


def plot_minority_f1_boxplots(qfl_client_results, baseline_client_results,
                              clients, client_train_loaders):
    sort_key = lambda cid: (cid.split("_sub")[0], int(cid.split("_sub")[1]))
    try:
        clients_sorted = sorted(clients, key=sort_key)
    except:
        clients_sorted = sorted(clients)

    num_clients = len(clients_sorted)
    nrows = int(np.ceil(num_clients / 2))
    ncols = 2 if num_clients > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).flatten() if num_clients > 1 else [axes]

    all_vals_global = []

    for idx, cid in enumerate(clients_sorted):
        ax = axes[idx]
        mcls = get_minority_class(client_train_loaders[cid])
        key = f"f1_class_{mcls}"

        vq = [qfl_client_results[s][cid].get(key, np.nan) for s in qfl_client_results]
        vb = [baseline_client_results[s][cid].get(key, np.nan) for s in baseline_client_results]

        vq = [x for x in vq if not np.isnan(x)]
        vb = [x for x in vb if not np.isnan(x)]
        all_vals_global.extend(vq + vb)

        if len(vq) == 0 and len(vb) == 0:
            ax.text(0.5, 0.5, "No data", ha="center")
            continue

        styled_boxplot(ax, [vq, vb], ["QFL", "Baseline"])

        ratio = get_minority_ratio(client_train_loaders[cid])
        ax.set_title(f"{cid}\n(Minority={mcls}, Ratio={ratio:.2f})")
        ax.set_ylabel("Minority F1")
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    if all_vals_global:
        lo = max(0, min(all_vals_global) - 0.05)
        hi = min(1, max(all_vals_global) + 0.05)
        for ax in axes: ax.set_ylim(lo, hi)

    for i in range(num_clients, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Minority-Class F1 under Label Distribution Skew", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_f1_vs_minority_ratio(qfl_client_results, baseline_client_results, client_train_loaders, seeds):
    cids = list(client_train_loaders.keys())
    x = [get_minority_ratio(client_train_loaders[c]) for c in cids]

    y_q_mean, y_q_std = [], []
    y_b_mean, y_b_std = [], []

    for cid in cids:
        mcls = get_minority_class(client_train_loaders[cid])
        key = f"f1_class_{mcls}"

        vals_q = [qfl_client_results[s][cid].get(key, np.nan) for s in seeds]
        vals_q = [v for v in vals_q if not np.isnan(v)]
        y_q_mean.append(np.mean(vals_q) if vals_q else np.nan)
        y_q_std.append(np.std(vals_q) if vals_q else 0)

        vals_b = [baseline_client_results[s][cid].get(key, np.nan) for s in seeds]
        vals_b = [v for v in vals_b if not np.isnan(v)]
        y_b_mean.append(np.mean(vals_b) if vals_b else np.nan)
        y_b_std.append(np.std(vals_b) if vals_b else 0)

    plt.figure(figsize=(7, 6))
    plt.errorbar(x, y_q_mean, yerr=y_q_std, fmt='o', label="QFL",
                 color="dodgerblue", alpha=0.7, capsize=4)
    plt.errorbar(x, y_b_mean, yerr=y_b_std, fmt='o', label="Baseline",
                 color="darkorange", alpha=0.7, capsize=4)
    plt.xlabel("Minority Ratio")
    plt.ylabel("Minority F1")
    plt.title("F1 vs. Minority Ratio (per Client, mean ± std over seeds)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.show()


def plot_minority_f1_density(qfl_client_results, baseline_client_results, client_train_loaders, seeds):
    vals_q, vals_b = [], []

    for cid in client_train_loaders:
        mcls = get_minority_class(client_train_loaders[cid])
        key = f"f1_class_{mcls}"

        for s in seeds:
            vq = qfl_client_results[s][cid].get(key, np.nan)
            vb = baseline_client_results[s][cid].get(key, np.nan)
            if not np.isnan(vq): vals_q.append(vq)
            if not np.isnan(vb): vals_b.append(vb)

    plt.figure(figsize=(7, 5))
    sns.kdeplot(vals_q, label="QFL", color="dodgerblue", fill=True, alpha=0.3)
    sns.kdeplot(vals_b, label="Baseline", color="darkorange", fill=True, alpha=0.3)
    plt.xlabel("F1 (Minority)")
    plt.title("Minority F1 Distribution (all seeds combined)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.show()


def plot_stability_boxplot(qfl_global_metrics, baseline_client_metrics, central_global_metrics):
    """Plottet Stabilität der Modelle über Seeds."""
    seeds = list(qfl_global_metrics.keys())

    qfl_vals = [qfl_global_metrics[s]["accuracy"] for s in seeds]
    qfl_mean = np.mean(qfl_vals)
    qfl_std = np.std(qfl_vals)

    baseline_means = []
    baseline_stds = []
    for s in seeds:
        client_accs = [v["accuracy"] for v in baseline_client_metrics[s].values()]
        baseline_means.append(np.mean(client_accs))
        baseline_stds.append(np.std(client_accs))
    baseline_mean = np.mean(baseline_means)
    baseline_std = np.sqrt(np.mean(np.array(baseline_stds) ** 2))

    central_vals = [central_global_metrics[s]["accuracy"] for s in seeds]
    central_mean = np.mean(central_vals)
    central_std = np.std(central_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ['QFL Global', 'Avg Local', 'Central']
    means = [qfl_mean, baseline_mean, central_mean]
    stds = [qfl_std, baseline_std, central_std]

    colors = ['dodgerblue', 'darkorange', 'gray']

    ax.bar(labels, means, yerr=stds, color=colors,
           alpha=0.8, capsize=6, edgecolor='black', linewidth=1.2)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Stability Across Seeds (mean ± std)", fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.show()


def plot_auc_heatmap_skewed(qfl_client_results, baseline_client_results, skewed_clients):
    if not skewed_clients: return
    deltas = {cid: [] for cid in skewed_clients}
    for s in qfl_client_results:
        for cid in skewed_clients:
            if cid in qfl_client_results[s]:
                d = qfl_client_results[s][cid]["auc"] - baseline_client_results[s][cid]["auc"]
                deltas[cid].append(d)
    data = [{"Client": cid, "Delta AUC": np.mean(vals)} for cid, vals in deltas.items() if vals]
    if not data: return
    df = pd.DataFrame(data).set_index("Client")
    plt.figure(figsize=(5, len(df) * 0.5 + 2))
    sns.heatmap(df, annot=True, cmap="RdBu", center=0, fmt=".3f", cbar_kws={'label': 'Gain'})
    plt.title("Avg Delta AUC (QFL - Baseline)")
    plt.tight_layout()
    plt.show()


def plot_auc_pr_heatmap_skewed(qfl_client_results, baseline_client_results, skewed_clients):
    if not skewed_clients: return
    deltas = {cid: [] for cid in skewed_clients}
    for s in qfl_client_results:
        for cid in skewed_clients:
            if cid in qfl_client_results[s]:
                val_q = qfl_client_results[s][cid].get("auc_pr", 0)
                val_b = baseline_client_results[s][cid].get("auc_pr", 0)
                deltas[cid].append(val_q - val_b)
    data = [{"Client": cid, "Delta AUC-PR": np.mean(vals)} for cid, vals in deltas.items() if vals]
    if not data: return
    df = pd.DataFrame(data).set_index("Client")
    plt.figure(figsize=(5, len(df) * 0.5 + 2))
    sns.heatmap(df, annot=True, cmap="RdBu", center=0, fmt=".3f", cbar_kws={'label': 'Gain'})
    plt.title("Avg Delta AUC-PR (QFL - Baseline)")
    plt.tight_layout()
    plt.show()


def plot_roc_pr_skewed(qfl_client_results, baseline_client_results, skewed_clients, seeds):
    """
    Plottet ROC und Precision-Recall Curves für skewed clients.
    Wichtig: PR-Curves sind bei Imbalance aussagekräftiger als ROC!
    """
    if not skewed_clients:
        print("No skewed clients to plot.")
        return

    num_clients = len(skewed_clients)
    fig, axes = plt.subplots(num_clients, 2, figsize=(14, 5 * num_clients))

    # Falls nur 1 skewed client, axes zu 2D array machen
    if num_clients == 1:
        axes = axes.reshape(1, -1)

    for idx, cid in enumerate(skewed_clients):
        ax_roc = axes[idx, 0]
        ax_pr = axes[idx, 1]

        # Sammle Predictions über alle Seeds
        qfl_probs_all = []
        qfl_labels_all = []
        base_probs_all = []
        base_labels_all = []

        for seed in seeds:
            if cid in qfl_client_results[seed]:
                qfl_probs = qfl_client_results[seed][cid].get('probs', [])
                qfl_labels = qfl_client_results[seed][cid].get('labels', [])

                base_probs = baseline_client_results[seed][cid].get('probs', [])
                base_labels = baseline_client_results[seed][cid].get('labels', [])

                if len(qfl_probs) > 0:
                    qfl_probs_all.extend(qfl_probs)
                    qfl_labels_all.extend(qfl_labels)
                    base_probs_all.extend(base_probs)
                    base_labels_all.extend(base_labels)

        if len(qfl_probs_all) == 0:
            ax_roc.text(0.5, 0.5, 'No probability data available',
                        ha='center', va='center', fontsize=12)
            ax_pr.text(0.5, 0.5, 'No probability data available',
                       ha='center', va='center', fontsize=12)
            continue

        # ROC Curve
        fpr_qfl, tpr_qfl, _ = roc_curve(qfl_labels_all, qfl_probs_all)
        fpr_base, tpr_base, _ = roc_curve(base_labels_all, base_probs_all)

        auc_qfl = auc(fpr_qfl, tpr_qfl)
        auc_base = auc(fpr_base, tpr_base)

        ax_roc.plot(fpr_qfl, tpr_qfl, label=f'QFL (AUC={auc_qfl:.3f})',
                    color='dodgerblue', linewidth=2)
        ax_roc.plot(fpr_base, tpr_base, label=f'Baseline (AUC={auc_base:.3f})',
                    color='darkorange', linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax_roc.set_xlabel('False Positive Rate', fontsize=11)
        ax_roc.set_ylabel('True Positive Rate', fontsize=11)
        ax_roc.set_title(f'{cid} - ROC Curve', fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, linestyle='--', alpha=0.3)

        # Precision-Recall Curve
        prec_qfl, rec_qfl, _ = precision_recall_curve(qfl_labels_all, qfl_probs_all)
        prec_base, rec_base, _ = precision_recall_curve(base_labels_all, base_probs_all)

        auc_pr_qfl = auc(rec_qfl, prec_qfl)
        auc_pr_base = auc(rec_base, prec_base)

        ax_pr.plot(rec_qfl, prec_qfl, label=f'QFL (AUC-PR={auc_pr_qfl:.3f})',
                   color='dodgerblue', linewidth=2)
        ax_pr.plot(rec_base, prec_base, label=f'Baseline (AUC-PR={auc_pr_base:.3f})',
                   color='darkorange', linewidth=2)

        # Baseline für PR Curve (Prevalence)
        prevalence = sum(qfl_labels_all) / len(qfl_labels_all)
        ax_pr.axhline(prevalence, color='k', linestyle='--', linewidth=1,
                      label=f'Random (Prev={prevalence:.3f})')

        ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=11)
        ax_pr.set_ylabel('Precision (PPV)', fontsize=11)
        ax_pr.set_title(f'{cid} - Precision-Recall Curve', fontweight='bold')
        ax_pr.legend(loc='lower left')
        ax_pr.grid(True, linestyle='--', alpha=0.3)

        # Print Summary
        print(f"\n{cid} Performance:")
        print(f"  QFL:      AUC-ROC={auc_qfl:.4f} | AUC-PR={auc_pr_qfl:.4f}")
        print(f"  Baseline: AUC-ROC={auc_base:.4f} | AUC-PR={auc_pr_base:.4f}")
        print(f"  Δ AUC-ROC: {auc_qfl - auc_base:+.4f}")
        print(f"  Δ AUC-PR:  {auc_pr_qfl - auc_pr_base:+.4f}")

    plt.tight_layout()
    plt.show()


def plot_roc_pr_comparison_all_clients(qfl_client_metrics, baseline_client_metrics,
                                       client_train_loaders, seeds):
    """
    Vergleicht ROC und PR Curves ALLER Clients in einem zusammenfassenden Plot.
    Färbt skewed clients anders ein.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))

    skewed_clients = identify_skewed_clients(client_train_loaders, threshold=0.3)

    for cid in client_train_loaders.keys():
        # Sammle Predictions über alle Seeds
        qfl_probs_all = []
        qfl_labels_all = []

        for seed in seeds:
            if cid in qfl_client_metrics[seed]:
                qfl_probs = qfl_client_metrics[seed][cid].get('probs', [])
                qfl_labels = qfl_client_metrics[seed][cid].get('labels', [])

                if len(qfl_probs) > 0:
                    qfl_probs_all.extend(qfl_probs)
                    qfl_labels_all.extend(qfl_labels)

        if len(qfl_probs_all) == 0:
            continue

        # Style basierend auf Skew
        is_skewed = cid in skewed_clients
        color = 'red' if is_skewed else 'blue'
        alpha = 0.8 if is_skewed else 0.3
        linewidth = 2 if is_skewed else 1

        # ROC Curve
        fpr, tpr, _ = roc_curve(qfl_labels_all, qfl_probs_all)
        roc_auc = auc(fpr, tpr)

        label = f'{cid} (AUC={roc_auc:.2f})' if is_skewed else None
        ax_roc.plot(fpr, tpr, color=color, alpha=alpha, linewidth=linewidth, label=label)

        # PR Curve
        prec, rec, _ = precision_recall_curve(qfl_labels_all, qfl_probs_all)
        pr_auc = auc(rec, prec)

        label_pr = f'{cid} (AUC-PR={pr_auc:.2f})' if is_skewed else None
        ax_pr.plot(rec, prec, color=color, alpha=alpha, linewidth=linewidth, label=label_pr)

    # ROC Plot finalisieren
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves - All Clients (QFL)\nRed = Skewed, Blue = Balanced',
                     fontweight='bold', fontsize=13)
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.grid(True, linestyle='--', alpha=0.3)

    # PR Plot finalisieren
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curves - All Clients (QFL)\nRed = Skewed, Blue = Balanced',
                    fontweight='bold', fontsize=13)
    ax_pr.legend(loc='lower left', fontsize=9)
    ax_pr.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_skewed(qfl_client_results, baseline_client_results, skewed_clients):
    if not skewed_clients: return
    seeds = list(qfl_client_results.keys())
    for cid in skewed_clients:
        cm_q = np.zeros((2, 2), dtype=int)
        cm_b = np.zeros((2, 2), dtype=int)
        count = 0
        for s in seeds:
            if cid in qfl_client_results[s]:
                cm_q += qfl_client_results[s][cid]["confusion_matrix"]
                cm_b += baseline_client_results[s][cid]["confusion_matrix"]
                count += 1
        if count == 0: continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(cm_q, display_labels=["Norm", "Pneu"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
        ax[0].set_title(f"{cid} QFL (Sum)")
        ConfusionMatrixDisplay(cm_b, display_labels=["Norm", "Pneu"]).plot(ax=ax[1], cmap="Oranges", colorbar=False)
        ax[1].set_title(f"{cid} Base (Sum)")
        plt.tight_layout()
        plt.show()


def compute_bias_summary(qfl_client_metrics, baseline_client_metrics, qfl_external_metrics,
                         central_global_metrics, seeds, client_train_loaders=None):
    """Erweiterte Bias/Fairness Analyse mit statistischen Tests."""
    records = []

    for seed in seeds:
        for cid in qfl_client_metrics[seed]:
            qfl_m = qfl_client_metrics[seed][cid]
            base_m = baseline_client_metrics[seed][cid]

            client_data = {
                "Seed": seed,
                "Client": cid,
                "QFL_ACC": qfl_m["accuracy"],
                "QFL_F1": qfl_m["f1"],
                "Baseline_ACC": base_m["accuracy"],
                "Baseline_F1": base_m["f1"],
                "ACC_Gain": qfl_m["accuracy"] - base_m["accuracy"],
                "F1_Gain": qfl_m["f1"] - base_m["f1"],
            }

            if client_train_loaders and cid in client_train_loaders:
                labels = get_labels_from_dataset(client_train_loaders[cid].dataset)
                counts = Counter(labels)
                n_minority = counts.get(1, 0)
                n_total = len(labels)
                client_data["Minority_Ratio"] = n_minority / max(n_total, 1)
            else:
                client_data["Minority_Ratio"] = None

            records.append(client_data)

    df = pd.DataFrame(records)

    # Statistische Analyse
    print("\n" + "=" * 80)
    print("📊 BIAS & FAIRNESS SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {len(df)} experiments ({len(seeds)} seeds × {len(df) // len(seeds)} clients)")
    print(f"\nOverall Performance:")
    print(
        f"  QFL:      ACC={df['QFL_ACC'].mean():.3f}±{df['QFL_ACC'].std():.3f} | F1={df['QFL_F1'].mean():.3f}±{df['QFL_F1'].std():.3f}")
    print(
        f"  Baseline: ACC={df['Baseline_ACC'].mean():.3f}±{df['Baseline_ACC'].std():.3f} | F1={df['Baseline_F1'].mean():.3f}±{df['Baseline_F1'].std():.3f}")
    print(
        f"  Gain:     ACC={df['ACC_Gain'].mean():.3f}±{df['ACC_Gain'].std():.3f} | F1={df['F1_Gain'].mean():.3f}±{df['F1_Gain'].std():.3f}")

    # Wilcoxon Test
    try:
        _, p_acc = wilcoxon(df['QFL_ACC'], df['Baseline_ACC'])
        _, p_f1 = wilcoxon(df['QFL_F1'], df['Baseline_F1'])
        print(f"\nStatistical Significance (Wilcoxon):")
        print(f"  Accuracy: p={p_acc:.4f} {'✓ Significant' if p_acc < 0.05 else '✗ Not significant'}")
        print(f"  F1:       p={p_f1:.4f} {'✓ Significant' if p_f1 < 0.05 else '✗ Not significant'}")
    except:
        print("\nStatistical test failed (likely identical distributions)")

    # Win Rate
    print(f"\nWin Rate (QFL > Baseline):")
    print(f"  Accuracy: {(df['ACC_Gain'] > 0).sum()}/{len(df)} ({100 * (df['ACC_Gain'] > 0).mean():.1f}%)")
    print(f"  F1:       {(df['F1_Gain'] > 0).sum()}/{len(df)} ({100 * (df['F1_Gain'] > 0).mean():.1f}%)")

    print("=" * 80 + "\n")
    print(df.head(12))

    return df


# ============================================================================
# NEUE LDS-SPEZIFISCHE FUNKTIONEN
# ============================================================================

def plot_skewed_vs_balanced_comparison(qfl_client_metrics, baseline_client_metrics,
                                       client_train_loaders, seeds, skew_threshold=0.3):
    """
    Vergleicht Performance zwischen stark skewed (<30% minority) und balanced Clients.
    KERNFRAGE: Profitieren skewed clients MEHR von QFL?
    """
    skewed_clients = []
    balanced_clients = []

    for cid, loader in client_train_loaders.items():
        labels = get_labels_from_dataset(loader.dataset)
        counts = Counter(labels)
        minority_ratio = min(counts.values()) / len(labels)

        if minority_ratio < skew_threshold:
            skewed_clients.append(cid)
        else:
            balanced_clients.append(cid)

    # F1 Scores sammeln
    data = []
    for seed in seeds:
        for cid in skewed_clients:
            data.append({
                'Seed': seed,
                'Client_Type': 'Skewed',
                'Client': cid,
                'QFL_F1': qfl_client_metrics[seed][cid]['f1'],
                'Baseline_F1': baseline_client_metrics[seed][cid]['f1'],
                'F1_Gain': qfl_client_metrics[seed][cid]['f1'] - baseline_client_metrics[seed][cid]['f1']
            })

        for cid in balanced_clients:
            data.append({
                'Seed': seed,
                'Client_Type': 'Balanced',
                'Client': cid,
                'QFL_F1': qfl_client_metrics[seed][cid]['f1'],
                'Baseline_F1': baseline_client_metrics[seed][cid]['f1'],
                'F1_Gain': qfl_client_metrics[seed][cid]['f1'] - baseline_client_metrics[seed][cid]['f1']
            })

    df = pd.DataFrame(data)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: F1 Comparison
    ax1 = axes[0]
    df_melted = df.melt(id_vars=['Client_Type', 'Seed'],
                        value_vars=['QFL_F1', 'Baseline_F1'],
                        var_name='Method', value_name='F1')

    sns.boxplot(data=df_melted, x='Client_Type', y='F1', hue='Method', ax=ax1)
    ax1.set_title('F1 Score: Skewed vs. Balanced Clients', fontweight='bold', fontsize=14)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.legend(title='Method')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: F1 Gain
    ax2 = axes[1]
    sns.boxplot(data=df, x='Client_Type', y='F1_Gain', ax=ax2, color='steelblue')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title('F1 Improvement (QFL - Baseline)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('ΔF1', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Plot 3: Per-Client F1 Gain
    ax3 = axes[2]
    df_gain_mean = df.groupby(['Client_Type', 'Client'])['F1_Gain'].mean().reset_index()
    df_gain_mean = df_gain_mean.sort_values('F1_Gain', ascending=False)

    colors = ['red' if ct == 'Skewed' else 'green' for ct in df_gain_mean['Client_Type']]
    ax3.barh(df_gain_mean['Client'], df_gain_mean['F1_Gain'], color=colors, alpha=0.7)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Average F1 Gain', fontsize=12)
    ax3.set_title('Per-Client F1 Improvement', fontweight='bold', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    # Statistical Analysis
    print("\n" + "=" * 80)
    print("📊 SKEWED vs. BALANCED CLIENTS ANALYSIS")
    print("=" * 80)

    skewed_gain = df[df['Client_Type'] == 'Skewed']['F1_Gain']
    balanced_gain = df[df['Client_Type'] == 'Balanced']['F1_Gain']

    print(f"\nSkewed Clients (n={len(skewed_clients)}):")
    print(f"  F1 Gain: {skewed_gain.mean():.4f} ± {skewed_gain.std():.4f}")
    print(f"  Win Rate: {(skewed_gain > 0).mean() * 100:.1f}%")

    print(f"\nBalanced Clients (n={len(balanced_clients)}):")
    print(f"  F1 Gain: {balanced_gain.mean():.4f} ± {balanced_gain.std():.4f}")
    print(f"  Win Rate: {(balanced_gain > 0).mean() * 100:.1f}%")

    # Mann-Whitney U Test
    stat, p = mannwhitneyu(skewed_gain, balanced_gain)
    print(f"\nMann-Whitney U Test:")
    print(f"  p-value: {p:.4f}")
    if p < 0.05:
        if skewed_gain.mean() > balanced_gain.mean():
            print("  ✓ Skewed clients benefit SIGNIFICANTLY MORE from QFL!")
        else:
            print("  ✓ Balanced clients benefit more (unexpected)")
    else:
        print("  ✗ No significant difference between groups")

    print("=" * 80 + "\n")

    return df


def plot_class_specific_performance(qfl_client_metrics, baseline_client_metrics,
                                    client_train_loaders, seeds):
    """
    Zeigt Performance für JEDE Klasse einzeln.
    Wichtig: Bei LDS leidet meist die Minority-Klasse.
    """
    data = []

    for seed in seeds:
        for cid, loader in client_train_loaders.items():
            labels = get_labels_from_dataset(loader.dataset)
            counts = Counter(labels)
            minority_class = 0 if counts.get(0, 0) < counts.get(1, 0) else 1
            majority_class = 1 - minority_class

            qfl_m = qfl_client_metrics[seed][cid]
            base_m = baseline_client_metrics[seed][cid]

            # Majority Class
            data.append({
                'Seed': seed,
                'Client': cid,
                'Class': 'Majority',
                'Class_Label': majority_class,
                'QFL_F1': qfl_m.get(f'f1_class_{majority_class}', np.nan),
                'Baseline_F1': base_m.get(f'f1_class_{majority_class}', np.nan),
                'Minority_Ratio': min(counts.values()) / len(labels)
            })

            # Minority Class
            data.append({
                'Seed': seed,
                'Client': cid,
                'Class': 'Minority',
                'Class_Label': minority_class,
                'QFL_F1': qfl_m.get(f'f1_class_{minority_class}', np.nan),
                'Baseline_F1': base_m.get(f'f1_class_{minority_class}', np.nan),
                'Minority_Ratio': min(counts.values()) / len(labels)
            })

    df = pd.DataFrame(data).dropna()
    df['F1_Gain'] = df['QFL_F1'] - df['Baseline_F1']

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: F1 Score by Class
    ax1 = axes[0]
    df_melted = df.melt(id_vars=['Class', 'Seed'],
                        value_vars=['QFL_F1', 'Baseline_F1'],
                        var_name='Method', value_name='F1')

    sns.violinplot(data=df_melted, x='Class', y='F1', hue='Method', ax=ax1, split=True)
    ax1.set_title('Class-Specific F1 Performance', fontweight='bold', fontsize=14)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Plot 2: F1 Gain by Class
    ax2 = axes[1]
    sns.boxplot(data=df, x='Class', y='F1_Gain', ax=ax2, palette='Set2')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('F1 Improvement by Class (QFL - Baseline)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('ΔF1', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Analysis
    print("\n" + "=" * 80)
    print("📊 CLASS-SPECIFIC PERFORMANCE ANALYSIS")
    print("=" * 80)

    for class_type in ['Majority', 'Minority']:
        subset = df[df['Class'] == class_type]
        print(f"\n{class_type} Class:")
        print(f"  QFL F1:      {subset['QFL_F1'].mean():.4f} ± {subset['QFL_F1'].std():.4f}")
        print(f"  Baseline F1: {subset['Baseline_F1'].mean():.4f} ± {subset['Baseline_F1'].std():.4f}")
        print(f"  F1 Gain:     {subset['F1_Gain'].mean():.4f} ± {subset['F1_Gain'].std():.4f}")

        # Wilcoxon Test
        try:
            _, p = wilcoxon(subset['QFL_F1'], subset['Baseline_F1'])
            print(f"  p-value:     {p:.4f} {'✓ Significant' if p < 0.05 else '✗ Not significant'}")
        except:
            print(f"  p-value:     Could not compute")

    print("=" * 80 + "\n")

    return df


def plot_skew_severity_analysis(qfl_client_metrics, baseline_client_metrics,
                                client_train_loaders, seeds):
    """
    Korrelation: Je stärker der Skew, desto mehr profitiert man von QFL?
    """
    data = []

    for cid, loader in client_train_loaders.items():
        labels = get_labels_from_dataset(loader.dataset)
        counts = Counter(labels)
        minority_ratio = min(counts.values()) / len(labels)

        # Skew Severity = Abweichung von 50%
        skew_severity = abs(0.5 - minority_ratio)

        for seed in seeds:
            qfl_f1 = qfl_client_metrics[seed][cid]['f1']
            base_f1 = baseline_client_metrics[seed][cid]['f1']

            data.append({
                'Client': cid,
                'Seed': seed,
                'Minority_Ratio': minority_ratio,
                'Skew_Severity': skew_severity,
                'F1_Gain': qfl_f1 - base_f1,
                'QFL_F1': qfl_f1,
                'Baseline_F1': base_f1
            })

    df = pd.DataFrame(data)

    # Correlation
    corr, p_value = spearmanr(df['Skew_Severity'], df['F1_Gain'])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter with Regression
    ax1 = axes[0]
    for seed in seeds:
        subset = df[df['Seed'] == seed]
        ax1.scatter(subset['Skew_Severity'], subset['F1_Gain'],
                    alpha=0.6, s=100, label=f'Seed {seed}')

    # Regression Line
    z = np.polyfit(df['Skew_Severity'], df['F1_Gain'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Skew_Severity'].min(), df['Skew_Severity'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (ρ={corr:.3f})')

    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Skew Severity (|0.5 - Minority Ratio|)', fontsize=12)
    ax1.set_ylabel('F1 Improvement (QFL - Baseline)', fontsize=12)
    ax1.set_title(f'Skew Severity vs. F1 Gain\n(Spearman ρ={corr:.3f}, p={p_value:.4f})',
                  fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Binned Analysis
    ax2 = axes[1]
    df['Skew_Bin'] = pd.cut(df['Skew_Severity'],
                            bins=[0, 0.1, 0.2, 0.3, 0.5],
                            labels=['Mild (<10%)', 'Moderate (10-20%)',
                                    'Severe (20-30%)', 'Extreme (>30%)'])

    sns.boxplot(data=df, x='Skew_Bin', y='F1_Gain', ax=ax2, palette='RdYlGn_r')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Skew Severity Category', fontsize=12)
    ax2.set_ylabel('F1 Improvement', fontsize=12)
    ax2.set_title('F1 Gain by Skew Severity', fontweight='bold', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Analysis
    print("\n" + "=" * 80)
    print("📊 SKEW SEVERITY vs. IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(f"\nSpearman Correlation: ρ = {corr:.4f}, p = {p_value:.4f}")

    if p_value < 0.05:
        if corr > 0:
            print("✓ POSITIVE correlation: Stronger skew → MORE benefit from QFL")
        else:
            print("✓ NEGATIVE correlation: Stronger skew → LESS benefit (unexpected)")
    else:
        print("✗ No significant correlation between skew severity and QFL benefit")

    print("\nF1 Gain by Severity:")
    for bin_name in df['Skew_Bin'].cat.categories:
        subset = df[df['Skew_Bin'] == bin_name]
        if len(subset) > 0:
            print(f"  {bin_name}: {subset['F1_Gain'].mean():.4f} ± {subset['F1_Gain'].std():.4f}")

    print("=" * 80 + "\n")

    return df


def analyze_lds_resilience(qfl_client_metrics, baseline_client_metrics,
                           client_train_loaders, seeds, skew_threshold=0.3):
    """
    Führt ALLE LDS-relevanten Analysen durch.
    Master-Funktion für vollständige Label Distribution Skew Analyse.

    Returns:
        dict mit allen Ergebnissen für Thesis-Tabellen
    """
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE LABEL DISTRIBUTION SKEW ANALYSIS")
    print("=" * 80)

    # 1. Skewed vs. Balanced
    print("\n[1/3] Analyzing Skewed vs. Balanced Clients...")
    df_comparison = plot_skewed_vs_balanced_comparison(
        qfl_client_metrics, baseline_client_metrics,
        client_train_loaders, seeds, skew_threshold
    )

    # 2. Class-Specific Performance
    print("\n[2/3] Analyzing Class-Specific Performance...")
    df_class = plot_class_specific_performance(
        qfl_client_metrics, baseline_client_metrics,
        client_train_loaders, seeds
    )

    # 3. Skew Severity vs. Improvement
    print("\n[3/3] Analyzing Skew Severity Impact...")
    df_severity = plot_skew_severity_analysis(
        qfl_client_metrics, baseline_client_metrics,
        client_train_loaders, seeds
    )

    print("\n" + "=" * 80)
    print("✅ COMPLETE LDS ANALYSIS FINISHED")
    print("=" * 80 + "\n")

    return {
        'comparison': df_comparison,
        'class_specific': df_class,
        'severity': df_severity
    }
