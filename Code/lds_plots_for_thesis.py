import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# 1. GLOBALE KONFIGURATION
# =============================================================================
# Definition der Farben für eine einheitliche Visualisierung
COLORS = {
    'Zentrale Baseline': '#59A14F',
    'Lokale Baselines': '#E15759',
    'QFL Global': '#4E79A7'
}

# Festlegung der Seeds und Pfade
SEEDS = [10, 31, 42, 52, 117]
BASE_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Thesis_Results"
SCENARIO = "REF_NOISELESS_L4"
# 5 Runden á 3 lokale Epochen ergeben 15 Epochen Gesamttraining
LOCAL_EPOCHS_PER_ROUND = 3


# =============================================================================
# 2. DATENLADUNG UND AGGREGATION
# =============================================================================
def load_scenario_data(base_path, seeds, scenario_name):
    """Lädt alle JSON-Dateien der Seeds und extrahiert die Metriken."""
    data_struct = {
        'history': {
            'cent_acc': [],
            'cent_loss': [],
            'local_acc': [],
            'local_loss': [],
            'qfl_acc': [],
            'qfl_loss': [],
            'qfl_f1_macro': []
        },
        'evaluation': {
            'raw_data': []
        }
    }

    for seed in seeds:
        file_path = os.path.join(base_path, scenario_name, f"S{seed}_full_results.json")

        if not os.path.exists(file_path):
            print(f"Warnung: Datei für Seed {seed} nicht gefunden")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)
            data_struct['evaluation']['raw_data'].append(data)

        # Extraktion der Trainingshistorie
        data_struct['history']['cent_acc'].append(data['history']['centralized']['train_acc'])
        data_struct['history']['cent_loss'].append(data['history']['centralized']['train_loss'])

        clients = list(data['history']['local_baselines'].keys())

        # Mittelwerte über alle 16 lokalen Clients pro Epoche
        l_accs = [data['history']['local_baselines'][c]['train_acc'] for c in clients]
        l_loss = [data['history']['local_baselines'][c]['train_loss'] for c in clients]
        data_struct['history']['local_acc'].append(np.mean(l_accs, axis=0))
        data_struct['history']['local_loss'].append(np.mean(l_loss, axis=0))

        # Mittelwerte der lokalen QFL-Modelle
        q_accs = [data['history']['qfl_local_history'][c]['train_acc'] for c in clients]
        q_loss = [data['history']['qfl_local_history'][c]['train_loss'] for c in clients]
        data_struct['history']['qfl_acc'].append(np.mean(q_accs, axis=0))
        data_struct['history']['qfl_loss'].append(np.mean(q_loss, axis=0))

        # Validierungs-Verlauf des globalen Modells
        data_struct['history']['qfl_f1_macro'].append([h['f1_macro'] for h in data['history']['qfl_global']])

    return data_struct


# =============================================================================
# 3. VISUALISIERUNGS-FUNKTIONEN
# =============================================================================

def plot_training_dynamics(ds):
    """Erstellt Liniendiagramme für Genauigkeit und Verlust."""
    epochs = np.arange(1, 16)
    plt.figure(figsize=(12, 5))

    # Linker Plot: Trainingsgenauigkeit
    plt.subplot(1, 2, 1)

    for key, label, color in [('cent_acc', 'Zentrale Baseline', COLORS['Zentrale Baseline']),
                              ('local_acc', 'Lokale Baseline', COLORS['Lokale Baselines']),
                              ('qfl_acc', 'QFL Lokal', COLORS['QFL Global'])]:
        m = np.mean(ds['history'][key], axis=0)
        s = np.std(ds['history'][key], axis=0)
        plt.plot(epochs, m, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, m - s, m + s, color=color, alpha=0.15)

    plt.title('Trainingsgenauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    plt.legend()

    # Rechter Plot: Trainingsverlust
    plt.subplot(1, 2, 2)
    for key, label, color in [('cent_loss', 'Zentrale Baseline', COLORS['Zentrale Baseline']),
                              ('local_loss', 'Lokale Baseline', COLORS['Lokale Baselines']),
                              ('qfl_loss', 'QFL Lokal', COLORS['QFL Global'])]:
        m = np.mean(ds['history'][key], axis=0)
        s = np.std(ds['history'][key], axis=0)
        plt.plot(epochs, m, label=label, color=color, linewidth=2)
        plt.fill_between(epochs, m - s, m + s, color=color, alpha=0.15)

    plt.title('Trainingsverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Loss (BCE)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('01_training_dynamics.png', dpi=300)
    plt.close()


def plot_f1_macro_evolution(ds):
    """Stellt den Fairness-Gewinn des globalen Modells dar."""
    all_data = ds['evaluation']['raw_data']
    f1_evol = np.array(ds['history']['qfl_f1_macro'])

    # Referenzwert der Isolation
    final_loc_f1 = np.mean(
        [np.mean([d['evaluation']['local'][c]['f1_macro'] for c in d['evaluation']['local']]) for d in all_data])

    x_epochs = np.arange(1, 6) * LOCAL_EPOCHS_PER_ROUND
    m = np.mean(f1_evol, axis=0)
    s = np.std(f1_evol, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(x_epochs, m, color=COLORS['QFL Global'], linewidth=2.5, marker='o', label='Globales QFL-Modell (Verlauf)')
    plt.fill_between(x_epochs, m - s, m + s, color=COLORS['QFL Global'], alpha=0.15)
    plt.axhline(y=final_loc_f1, color=COLORS['Lokale Baselines'], linestyle='--', linewidth=2,
                label='Ø Lokale Baseline (Finale Isolation)')

    plt.xlabel('Kumulierte Trainings-Epochen')
    plt.ylabel('F1-Macro Score')
    plt.title('Fairness-Fortschritt: Globaler Verbund vs. Lokale Isolation')
    plt.xticks(x_epochs)
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('02_f1_macro_evolution.png', dpi=300)
    plt.close()


def plot_lds_stress_test(ds):
    """Analyse der Performance bei den 10/90 Bias-Clients."""
    all_data = ds['evaluation']['raw_data']
    targets = [('client_1_sub_1', 'C1.1 (Z1 - 10/90)'), ('client_4_sub_1', 'C4.1 (Z4 - 90/10)')]

    l_bal = [np.mean([d['evaluation']['local'][t[0]]['balanced_accuracy'] for d in all_data]) for t in targets]
    q_bal = [np.mean([d['evaluation']['qfl_global'][t[0]]['balanced_accuracy'] for d in all_data]) for t in targets]
    l_rec = [np.mean([d['evaluation']['local'][t[0]]['recall_minority'] for d in all_data]) for t in targets]
    q_rec = [np.mean([d['evaluation']['qfl_global'][t[0]]['recall_minority'] for d in all_data]) for t in targets]

    x = np.arange(len(targets))
    width = 0.35
    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, l_bal, width, label='Isolation (Balanced Acc)', color=COLORS['Lokale Baselines'], alpha=0.4)
    plt.bar(x + width / 2, q_bal, width, label='QFL Global (Balanced Acc)', color=COLORS['QFL Global'], alpha=0.4)
    plt.bar(x - width / 2, l_rec, width * 0.6, label='Isolation (Recall Minderheit)', color=COLORS['Lokale Baselines'])
    plt.bar(x + width / 2, q_rec, width * 0.6, label='QFL Global (Recall Minderheit)', color=COLORS['QFL Global'])

    plt.xticks(x, [t[1] for t in targets])
    plt.ylabel('Metrik-Wert')
    plt.ylim(0, 1.1)
    plt.title('LDS Stress-Test: Balanced Acc & Recall der Minderheitsklasse')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('03_lds_stress_test.png', dpi=300)
    plt.close()


def plot_robustness_analysis(ds):
    """Vergleicht ROC-AUC und durchschnittliche Konfidenz für Stress-Clients."""
    all_data = ds['evaluation']['raw_data']
    targets = [('client_1_sub_1', 'C1.1 (Z1)'), ('client_4_sub_1', 'C4.1 (Z4)')]

    metrics = ['roc_auc_global', 'avg_confidence']
    titles = ['ROC-AUC (Diskriminierungsgüte)', 'Ø Modellsicherheit (Konfidenz)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, metric in enumerate(metrics):
        l_vals = [np.mean([d['evaluation']['local'][t[0]][metric] for d in all_data]) for t in targets]
        q_vals = [np.mean([d['evaluation']['qfl_global'][t[0]][metric] for d in all_data]) for t in targets]

        x = np.arange(len(targets))
        width = 0.35

        axes[idx].bar(x - width / 2, l_vals, width, label='Lokale Baseline', color=COLORS['Lokale Baselines'],
                      alpha=0.8)
        axes[idx].bar(x + width / 2, q_vals, width, label='QFL Global', color=COLORS['QFL Global'], alpha=0.8)

        axes[idx].set_ylabel('Wert')
        axes[idx].set_title(titles[idx])
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([t[1] for t in targets])
        axes[idx].set_ylim(0, 1.1)
        axes[idx].legend()

    plt.suptitle('Robustheits-Analyse: Schwellenwert-unabhängige Güte und Vertrauen')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('05_robustness_analysis.png', dpi=300)
    plt.close()


def plot_normalized_cm_final(ds):
    """Konfusionsmatrizen mit gemittelten absoluten Zahlen über Seeds."""
    all_data = ds['evaluation']['raw_data']
    targets = [
        ('client_1_sub_1', 'Subclient 1.1 (Z1 - 10/90 N/P)'),
        ('client_4_sub_1', 'Subclient 4.1 (Z4 - 90/10 N/P)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for i, (cid, name) in enumerate(targets):
        for j, mode in enumerate(['local', 'qfl_global']):
            tp_list = [d['evaluation'][mode][cid]['true_positives'] for d in all_data]
            tn_list = [d['evaluation'][mode][cid]['true_negatives'] for d in all_data]
            fp_list = [d['evaluation'][mode][cid]['false_positives'] for d in all_data]
            fn_list = [d['evaluation'][mode][cid]['false_negatives'] for d in all_data]

            # Mittelwerte der absoluten Zahlen berechnen
            cm = np.array([
                [np.mean(tn_list), np.mean(fp_list)],
                [np.mean(fn_list), np.mean(tp_list)]
            ], dtype=float)

            sns.heatmap(
                cm,
                annot=True,
                fmt='.1f',
                cmap='Blues' if j == 1 else 'Reds',
                ax=axes[i, j],
                cbar=False
            )

            title_prefix = 'Lokale Isolation' if j == 0 else 'Globales QFL-Modell'
            axes[i, j].set_title(f"{title_prefix}\n{name}", fontsize=12)
            axes[i, j].set_xticklabels(['Normal', 'Pneumonie'])
            axes[i, j].set_yticklabels(['Normal', 'Pneumonie'])
            axes[i, j].set_xlabel('Vorhersage')
            axes[i, j].set_ylabel('Wahrheit')

    plt.tight_layout()
    plt.savefig('04_konfusionsmatrizen_final.png', dpi=300)
    plt.close()


# =============================================================================
# 4. HAUPTPROGRAMM
# =============================================================================
if __name__ == "__main__":
    results = load_scenario_data(BASE_PATH, SEEDS, SCENARIO)

    if results['evaluation']['raw_data']:
        plot_training_dynamics(results)
        plot_f1_macro_evolution(results)
        plot_lds_stress_test(results)
        plot_normalized_cm_final(results)
        plot_robustness_analysis(results)
        print("Auswertung abgeschlossen. Alle 5 Plots wurden im Verzeichnis gespeichert.")
    else:
        print("Fehler: Daten konnten nicht geladen werden.")