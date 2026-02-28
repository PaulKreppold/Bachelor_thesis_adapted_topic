import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import os

# Globale Konfiguration
COLORS = {
    'Zentrale Baseline': '#59A14F',
    'Lokale Baseline': '#E15759',
    'QFL Global': '#4E79A7',
    'QFL Lokal': '#F28E2B',
}

# Pfade und Seeds
SEEDS = [10, 31, 42, 52, 117]
BASE_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Thesis_Results"
SCENARIO = "REF_NOISELESS_L4"

MODEL_KEYS = {
    'Zentrale Baseline': 'centralized',
    'Lokale Baseline': 'local',
    'QFL Global': 'qfl_global',
    'QFL Lokal': 'qfl_local',
}

CLIENT_GROUPS = ['client_1', 'client_2', 'client_3', 'client_4']

CLIENT_META = {
    'client_1': {'dataset': 'PneumoniaMNIST', 'minority': 'Normal'},
    'client_2': {'dataset': 'RSNA', 'minority': 'Pneumonie'},
    'client_3': {'dataset': 'RSNA', 'minority': 'Normal'},
    'client_4': {'dataset': 'CheXpert', 'minority': 'Pneumonie'},
}

EXTREME_LDS_CLIENTS = ['client_1_sub_1', 'client_4_sub_1']


def format_cid(cid):
    # Formatiert technische IDs in Kurzform um
    return cid.replace('client_', 'C').replace('_sub_', '.')


# Datenverarbeitung
def load_scenario_data(base_path, seeds, scenario_name):
    # Aggregiert Ergebnisse über verschiedene Seeds
    ds = {'history': {k: [] for k in ['cent_acc', 'cent_loss', 'local_acc', 'local_loss', 'qfl_acc', 'qfl_loss']},
          'evaluation': {'raw_data': []}}
    for seed in seeds:
        path = os.path.join(base_path, scenario_name, f"S{seed}_full_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        ds['evaluation']['raw_data'].append(data)
        cl = list(data['history']['local_baselines'].keys())
        ds['history']['cent_acc'].append(data['history']['centralized']['train_acc'])
        ds['history']['cent_loss'].append(data['history']['centralized']['train_loss'])
        ds['history']['local_acc'].append(
            np.mean([data['history']['local_baselines'][c]['train_acc'] for c in cl], axis=0))
        ds['history']['local_loss'].append(
            np.mean([data['history']['local_baselines'][c]['train_loss'] for c in cl], axis=0))
        ds['history']['qfl_acc'].append(
            np.mean([data['history']['qfl_local_history'][c]['train_acc'] for c in cl], axis=0))
        ds['history']['qfl_loss'].append(
            np.mean([data['history']['qfl_local_history'][c]['train_loss'] for c in cl], axis=0))
    return ds


def extract_metric(raw_data_list, metric):
    # Extrahiert Metriken für die Evaluation
    result = {mk: {} for mk in MODEL_KEYS.values()}
    for data in raw_data_list:
        for mk in MODEL_KEYS.values():
            if mk in data['evaluation']:
                for client, vals in data['evaluation'][mk].items():
                    result[mk].setdefault(client, []).append(vals[metric])
    return result


# Plotting Funktionen
def plot_training_dynamics(ds, output_path='01_training_dynamics.png'):
    # Visualisierung der Epochen-Verläufe
    epochs = np.arange(1, 16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [(axes[0], [('cent_acc', 'Zentrale Baseline'), ('local_acc', 'Lokale Baseline'), ('qfl_acc', 'QFL Global')],
              'Trainingsgenauigkeit', 'Genauigkeit'),
             (axes[1],
              [('cent_loss', 'Zentrale Baseline'), ('local_loss', 'Lokale Baseline'), ('qfl_loss', 'QFL Global')],
              'Trainingsverlust', 'Verlust')]
    for ax, series, title, ylabel in specs:
        for key, label in series:
            if not ds['history'][key]:
                continue
            m, s = np.mean(ds['history'][key], axis=0), np.std(ds['history'][key], axis=0)
            ax.plot(epochs, m, label=label, color=COLORS[label], linewidth=2)
            ax.fill_between(epochs, m - s, m + s, color=COLORS[label], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel('Epoche')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


LDS_PANELS = [
    {'label': 'C1.1', 'minority': 'Normal', 'clients': ['client_1_sub_1']},
    {'label': 'C1.2–4', 'minority': 'Normal', 'clients': ['client_1_sub_2', 'client_1_sub_3', 'client_1_sub_4']},
    {'label': 'C2.1–4', 'minority': 'Pneumonie', 'clients': [f'client_2_sub_{i}' for i in range(1, 5)]},
    {'label': 'C3.1–4', 'minority': 'Normal', 'clients': [f'client_3_sub_{i}' for i in range(1, 5)]},
    {'label': 'C4.1', 'minority': 'Pneumonie', 'clients': ['client_4_sub_1']},
    {'label': 'C4.2–4', 'minority': 'Pneumonie', 'clients': ['client_4_sub_2', 'client_4_sub_3', 'client_4_sub_4']},
]


def plot_lds_f1_minority_per_group(ds, output_path='03_lds_f1_minority_per_group.png'):
    # Balkendiagramm F1-Score Minderheit
    raw_data = ds['evaluation']['raw_data']
    metric_data = extract_metric(raw_data, 'f1_minority')
    n_panels, n_models = len(LDS_PANELS), len(MODEL_KEYS)
    bar_width, x = 0.16, np.arange(n_panels)
    fig, ax = plt.subplots(figsize=(15, 7.0))

    bar_means = {}
    for i, (label, mk) in enumerate(MODEL_KEYS.items()):
        means, errs = [], []
        for pi, panel in enumerate(LDS_PANELS):
            vals = np.array([v for c in panel['clients'] for v in metric_data[mk].get(c, [])])
            m = np.mean(vals) if len(vals) else 0.0
            means.append(m)
            errs.append(np.std(vals) if len(vals) else 0.0)
            bar_means[(i, pi)] = m
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, label=label, color=COLORS[label], yerr=errs, capsize=3, alpha=0.85,
               edgecolor='white')
        for pi in range(n_panels):
            if bar_means[(i, pi)] == 0.0:
                ax.text(pi + offset, 0.012, '0.0', ha='center', va='bottom', fontsize=7.5, color=COLORS[label],
                        fontweight='bold')

    for sep_x in [1.5, 3.5]:
        ax.axvline(sep_x, color='dimgray', linewidth=1.2, linestyle='--', alpha=0.5)

    groups = [(0, 1, 'Client 1: PneumoniaMNIST'), (2, 3, 'Client 2 / 3: RSNA'), (4, 5, 'Client 4: CheXpert')]
    for s, e, lbl in groups:
        ax.text((s + e) / 2, -0.25, lbl, ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p['label']}\n[Minderheit: {p['minority']}]" for p in LDS_PANELS], fontsize=9)
    ax.set_ylabel('F1-Score (Minderheitsklasse)')
    ax.legend(loc='upper right')
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lds_performance_gap_heatmap(ds, output_path='04_lds_performance_gap_heatmap.png'):
    # Heatmap zur Analyse der Leistungslücke
    raw_data = ds['evaluation']['raw_data']
    if not raw_data:
        return

    metric_data = extract_metric(raw_data, 'performance_gap')
    clients = sorted(list(metric_data['centralized'].keys()))
    model_labels = list(MODEL_KEYS.keys())

    matrix = np.zeros((len(model_labels), len(clients)))
    for i, (_, mk) in enumerate(MODEL_KEYS.items()):
        for j, client in enumerate(clients):
            vals = metric_data[mk].get(client, [0])
            matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=0.75)

    ax.set_xticks(range(len(clients)))
    short = [c.replace('client_', 'C').replace('_sub_', '.') for c in clients]
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(model_labels)):
        for j in range(len(clients)):
            v = matrix[i, j]
            tc = 'white' if v > 0.45 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5, color=tc)

    for sep in [3.5, 7.5, 11.5]:
        ax.axvline(sep, color='black', linewidth=1.5, linestyle=':', alpha=0.7)

    for extreme_idx in [0, 12]:
        ax.add_patch(plt.Rectangle(
            (extreme_idx - 0.5, -0.5), 1, len(model_labels),
            fill=False, edgecolor='black', linewidth=2.5, clip_on=False
        ))

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Performance Gap (F1 Majority - F1 Minority)', fontsize=9)

    group_mids = [1.5, 5.5, 9.5, 13.5]
    for gi, (mid, group) in enumerate(zip(group_mids, CLIENT_GROUPS)):
        meta = CLIENT_META[group]
        ax.text(mid, -0.35,
                f"C{gi + 1} ({meta['dataset']})\nMinority: {meta['minority']}",
                ha='center', va='top', fontsize=8, color='dimgray',
                transform=ax.get_xaxis_transform())

    ax.set_title('LDS-Bias Heatmap: Performance Gap pro Sub-Client und Modell')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lds_extreme_clients_comparison(ds, output_path='05_lds_extreme_clients.png'):
    # Vergleich extremer LDS-Clients
    raw_data = ds['evaluation']['raw_data']
    metrics = [('f1_minority', 'F1-Score (Minderheit)'), ('recall_minority', 'Recall (Minderheit)'),
               ('balanced_accuracy', 'Balancierte Genauigkeit'), ('roc_auc_global', 'ROC-AUC')]
    extreme_clients = {'C1.1 (90% Pneu.)': 'client_1_sub_1', 'C4.1 (10% Pneu.)': 'client_4_sub_1'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    bar_width, x = 0.16, np.arange(len(metrics))
    for ax, (title, cid) in zip(axes, extreme_clients.items()):
        for i, (label, mk) in enumerate(MODEL_KEYS.items()):
            means = [np.mean(extract_metric(raw_data, m[0])[mk].get(cid, [0])) for m in metrics]
            offset = (i - 2 + 0.5) * bar_width
            ax.bar(x + offset, means, bar_width, label=label, color=COLORS[label], edgecolor='white')
            for mi, val in enumerate(means):
                if val == 0.0:
                    ax.text(x[mi] + offset, 0.012, '0.0', ha='center', va='bottom', fontsize=7.5,
                            color=COLORS[label], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics])
        ax.set_title(title)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lds_qfl_benefit_scatter(ds, output_path='06_lds_qfl_benefit_scatter.png'):
    # Scatter-Plot zum QFL-Nutzen
    raw_data = ds['evaluation']['raw_data']
    if not raw_data:
        return

    pg_data = extract_metric(raw_data, 'performance_gap')
    ba_data = extract_metric(raw_data, 'balanced_accuracy')
    group_colors = {'client_1': '#E15759', 'client_2': '#F28E2B', 'client_3': '#59A14F', 'client_4': '#4E79A7'}

    all_clients = sorted(list(pg_data['centralized'].keys()))
    x_v, y_v, cols, lbls = [], [], [], []
    for c in all_clients:
        x_val = np.mean(pg_data['local'].get(c, [0]))
        y_val = np.mean(ba_data['qfl_global'].get(c, [0])) - np.mean(ba_data['local'].get(c, [0]))
        x_v.append(x_val)
        y_v.append(y_val)
        cols.append(group_colors.get('_'.join(c.split('_')[:2]), 'gray'))
        lbls.append(format_cid(c))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_v, y_v, c=cols, s=110, edgecolors='white', alpha=0.9, zorder=5)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)

    for xi, yi, lbl in zip(x_v, y_v, lbls):
        ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(7, 0),
                    va='center', fontsize=8.5)

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(r'Verbesserung durch QFL $\uparrow$', xy=(0.98, 0), xycoords=trans,
                xytext=(0, 5), textcoords="offset points", ha='right', va='bottom', fontsize=9.5, color='#2266CC',
                fontweight='bold')
    ax.annotate(r'Verschlechterung durch QFL $\downarrow$', xy=(0.98, 0), xycoords=trans,
                xytext=(0, -5), textcoords="offset points", ha='right', va='top', fontsize=9.5, color='#CC2222',
                fontweight='bold')

    ax.set_xlabel('LDS-Intensität (Performance Gap)')
    ax.set_ylabel(r'QFL-Nutzen ($\Delta$ Balancierte Genauigkeit)')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lds_recall_minority_grouped(ds, output_path='07_lds_recall_minority_grouped.png'):
    # Boxplots zum Recall
    raw_data = ds['evaluation']['raw_data']
    recall_data = extract_metric(raw_data, 'recall_minority')
    fig, axes = plt.subplots(1, 6, figsize=(20, 6.5), sharey=True)
    for pi, (panel, ax) in enumerate(zip(LDS_PANELS, axes)):
        box_data = [[v for c in panel['clients'] for v in recall_data[mk].get(c, [0])] for mk in MODEL_KEYS.values()]
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.52)
        for patch, col in zip(bp['boxes'], COLORS.values()):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax.set_xticks(np.arange(1, 5))
        ax.set_xticklabels([l.replace(' ', '\n') for l in MODEL_KEYS.keys()], fontsize=7)
        ax.set_title(panel['label'], fontsize=9)
    axes[0].set_ylabel('Recall (Minderheitsklasse)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lds_fairness_radar(ds, output_path='08_lds_fairness_radar.png'):
    # Radar-Charts für Fairness-Metriken
    raw_data = ds['evaluation']['raw_data']
    METRICS = [('balanced_accuracy', 'Balancierte\nGenauigkeit'), ('f1_minority', 'F1\n(Minderheit)'),
               ('recall_minority', 'Recall\n(Minderheit)'), ('f1_macro', 'F1\n(Makro)'), ('roc_auc_global', 'ROC-AUC')]
    all_cl = sorted(list(raw_data[0]['evaluation']['centralized'].keys()))
    panels = [{'t': 'Moderater LDS',
               'c': [c for c in all_cl if ('client_1' in c or 'client_4' in c) and c not in EXTREME_LDS_CLIENTS]},
              {'t': 'Extremer LDS', 'c': EXTREME_LDS_CLIENTS},
              {'t': 'Referenz (RSNA)', 'c': [c for c in all_cl if 'client_2' in c or 'client_3' in c]}]
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist() + [0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), subplot_kw=dict(polar=True))

    for ax, p in zip(axes, panels):
        for lbl, mk in MODEL_KEYS.items():
            means = [
                np.mean([d['evaluation'][mk][c][m[0]] for d in raw_data for c in p['c'] if c in d['evaluation'][mk]])
                for m in METRICS]
            ax.plot(angles, means + [means[0]], label=lbl, color=COLORS[lbl], linewidth=2.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in METRICS], fontsize=9)
        ax.set_title(p['t'], pad=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(MODEL_KEYS), fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Hauptprogramm
if __name__ == '__main__':
    ds_struct = load_scenario_data(BASE_PATH, SEEDS, SCENARIO)
    if ds_struct['evaluation']['raw_data']:
        plot_training_dynamics(ds_struct)
        plot_lds_f1_minority_per_group(ds_struct)
        plot_lds_performance_gap_heatmap(ds_struct)
        plot_lds_extreme_clients_comparison(ds_struct)
        plot_lds_qfl_benefit_scatter(ds_struct)
        plot_lds_recall_minority_grouped(ds_struct)
        plot_lds_fairness_radar(ds_struct)
        print("Grafiken wurden erfolgreich erstellt.")