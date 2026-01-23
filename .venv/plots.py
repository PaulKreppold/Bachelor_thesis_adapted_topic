import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

GLOBAL_PALETTE = {
    "Lokale Baseline": "#D55E00",
    "QFL": "#0072B2",
    "Zentrale Baseline": "#009E73"
}

NOISE_COLOR_MAP = {
    "bitflip": "#E69F00",
    "phaseflip": "#56B4E9",
    "depolarizing": "#CC79A7",
    "None": "#000000"
}

SCENARIO_MAP = {
    'Zentrale Baseline': 'Zentrale Baseline',
    'Lokale Baseline': 'Lokale Baseline',
    'QFL': 'QFL'
}

LABEL_MAP = {
    'client_1': 'PneumoniaMNIST',
    'client_2': 'RSNA (60/40)',
    'client_3': 'RSNA (40/60)',
    'client_4': 'CheXpert (25/75)'
}

def setup_german_plot_style():
    sns.set_theme(style="white")
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'font.family': 'sans-serif',
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0
    })

def format_y_axis(ax, precision=2):
    #Fixiert die Nachkommastellen der Y-Achse
    formatter = FormatStrFormatter(f'%.{precision}f')
    ax.yaxis.set_major_formatter(formatter)

def apply_clear_boxplot_style(ax):
    #Erzeugt transparente Boxen mit farbigen Rändern
    for i, artist in enumerate(ax.patches):
        col = artist.get_facecolor()
        artist.set_facecolor('none')
        artist.set_edgecolor(col)
        artist.set_linewidth(1.5)

def get_noise_label(noise_type, p):
    #Erstellt eine saubere Beschriftung für Rauschszenarien
    nt = "Depolarizing Kanal" if noise_type == "depolarizing" else noise_type
    if str(noise_type) == "None" or p == 0:
        return "Szenario ohne Rauschen"
    return f"Rauschen: {nt} (p={p})"

# 1. TRAININGS- & KONVERGENZANALYSE

def plot_training_comparison(histories_baseline, histories_qfl_high_res, title, save_path):
    #Plottet den Mittelwert der Trainingsphase über alle Teilnehmer und Seeds
    setup_german_plot_style()
    plt.figure(figsize=(14, 7))
    metrics = [
        ('acc', 'Trainingsgenauigkeit', 1),
        ('loss', 'Trainingsverlust', 2)
    ]
    for key, label, idx in metrics:
        ax = plt.subplot(1, 2, idx)
        if histories_qfl_high_res:
            all_curves = [h[f'train_{key}'] for h in histories_qfl_high_res]
            max_e = max(len(c) for c in all_curves)
            matrix = np.full((len(all_curves), max_e), np.nan)
            for i, c in enumerate(all_curves):
                matrix[i, :len(c)] = c
            mean_curve = np.nanmean(matrix, axis=0)
            plt.plot(
                np.arange(1, max_e + 1),
                mean_curve,
                label='QFL (Ø über alle Teilnehmer)',
                color=GLOBAL_PALETTE['QFL'],
                lw=2
            )
        if histories_baseline:
            all_b = [h[f'train_{key}'] for h in histories_baseline]
            max_b = max(len(c) for c in all_b)
            matrix_b = np.full((len(all_b), max_b), np.nan)
            for i, c in enumerate(all_b):
                matrix_b[i, :len(c)] = c
            mean_b = np.nanmean(matrix_b, axis=0)
            plt.plot(
                np.arange(1, max_b + 1),
                mean_b,
                label='Lokale Baseline',
                color=GLOBAL_PALETTE['Lokale Baseline'],
                lw=1.5,
                alpha=0.7
            )
        ax.set_title(f'Verlauf: {label}', fontweight='bold')
        ax.set_xlabel('Epochen')
        ax.set_ylabel(label)
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':', alpha=0.5)
        format_y_axis(ax)
    plt.suptitle(f"Trainingsverlauf ({title})", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_convergence_speed_comparison(histories_baseline, histories_qfl, save_dir):
    #Analysiert die benötigten Epochen bis zum Erreichen von 95% der Endgenauigkeit
    setup_german_plot_style()
    convergence_data = []
    def get_conv_epoch(history, key='train_acc'):
        curve = history.get(key, [])
        if len(curve) == 0:
            return None
        final_val = curve[-1]
        threshold = 0.95 * final_val
        for i, val in enumerate(curve):
            if val >= threshold:
                return i + 1
        return len(curve)
    for h in histories_qfl:
        ep = get_conv_epoch(h, 'train_acc')
        if ep:
            convergence_data.append({'Szenario': 'QFL', 'Epochen': ep})
    for h in histories_baseline:
        ep = get_conv_epoch(h, 'train_acc')
        if ep:
            convergence_data.append({'Szenario': 'Lokale Baseline', 'Epochen': ep})
    df = pd.DataFrame(convergence_data)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df, x='Szenario', y='Epochen',
        palette=GLOBAL_PALETTE, width=0.4
    )
    ax.set_title("Verzögerung der Konvergenzgeschwindigkeit durch Rauschen", fontweight='bold')
    ax.set_ylabel("Epochen bis zum Erreichen von 95% der Endgenauigkeit")
    ax.set_xlabel("Untersuchtes Szenario")
    format_y_axis(ax, precision=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Konvergenz_Verzoegerung_Boxplot.png"), dpi=300)
    plt.close()

# 2. GLOBALER PERFORMANCE-VERGLEICH

def plot_global_comparison_summary(df, save_dir):
    #Boxplot für den globalen Vergleich der Testgenauigkeit
    setup_german_plot_style()
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df, x='Szenario', y='Testgenauigkeit', hue='Szenario',
        legend=False, order=["Zentrale Baseline", "Lokale Baseline", "QFL"],
        palette=GLOBAL_PALETTE, width=0.35, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    apply_clear_boxplot_style(ax)
    noise_label = get_noise_label(df['Noise_Type'].iloc[0], df['Noise_P'].iloc[0])
    ax.set_title(f"Globaler Vergleich der Testgenauigkeit\n({noise_label})")
    ax.set_ylabel("Testgenauigkeit")
    format_y_axis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Global_Summary_Boxplot.png"), dpi=300)
    plt.close()

def plot_paper_style_boxplot(df, save_dir):
    #Aufgeschlüsselter Boxplot nach klinischen Domänen
    setup_german_plot_style()
    df_plot = df.copy()
    df_plot['Domäne'] = df_plot['Client'].map(LABEL_MAP)
    plt.figure(figsize=(13, 7))
    ax = sns.boxplot(
        data=df_plot, x='Domäne', y='Testgenauigkeit', hue='Szenario',
        palette=GLOBAL_PALETTE, width=0.55, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    apply_clear_boxplot_style(ax)
    noise_label = get_noise_label(df['Noise_Type'].iloc[0], df['Noise_P'].iloc[0])
    ax.set_title(f"Domänenspezifische Robustheit der Testgenauigkeit ({noise_label})")
    ax.set_ylabel("Testgenauigkeit")
    plt.legend(frameon=True, loc='lower right', edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Detailliert_Performance_Boxplot.png"), dpi=300)
    plt.close()

# 3. LDS RESCUE & ROBUSTHEITS-MATRIZEN

def plot_recall_improvement_lds(df, save_dir):
    #Visualisiert die Verbesserung des Recall der Minderheitsklasse durch QFL
    setup_german_plot_style()
    targets = [('client_1', '1'), ('client_4', '1')]
    df_lds = df[df.apply(lambda x: (str(x['Client']), str(x['Subclient'])) in targets, axis=1)].copy()
    if df_lds.empty:
        return
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_lds, x='Client', y='Recall_Minderheit', hue='Szenario',
        palette=GLOBAL_PALETTE, width=0.65, capsize=.05
    )
    noise_label = get_noise_label(df['Noise_Type'].iloc[0], df['Noise_P'].iloc[0])
    ax.set_title(f"Minderheiten-Resilienz (LDS-Rettung)\n{noise_label}")
    ax.set_ylabel("Recall der Minderheitsklasse")
    ax.set_ylim(0, 1.0)
    format_y_axis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "LDS_Rescue_Recall.png"), dpi=300)
    plt.close()

def plot_noise_rescue_erosion(df_master, save_dir):
    """Analysiert den Verlust des QFL-Vorteils bei steigendem Rauschen."""
    setup_german_plot_style()
    targets = [('client_1', '1'), ('client_4', '1')]
    df_ext = df_master[df_master.apply(lambda x: (str(x['Client']), str(x['Subclient'])) in targets, axis=1)].copy()
    qfl = df_ext[df_ext['Szenario'] == 'QFL']
    loc = df_ext[df_ext['Szenario'] == 'Lokale Baseline']
    merged = pd.merge(qfl, local, on=['Seed', 'Noise_Type', 'Noise_P', 'Client'], suffixes=('_qfl', '_loc'))
    merged['Vorteil'] = merged['Recall_Minderheit_qfl'] - merged['Recall_Minderheit_loc']
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=merged, x='Noise_P', y='Vorteil', hue='Noise_Type', marker='o', palette=NOISE_COLOR_MAP)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Erosion des QFL-Vorteils (LDS-Rettung) unter Rauschen")
    plt.xlabel("Rauschwahrscheinlichkeit p")
    plt.ylabel("Differenz Recall (QFL - Lokale Baseline)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Master_LDS_Rescue_Erosion.png"), dpi=300)
    plt.close()

def plot_skew_noise_heatmap(df_master, save_dir):
    #Visualisiert das Zusammenspiel von Rauschen und LDS-Heterogenität
    setup_german_plot_style()
    df_qfl = df_master[df_master['Szenario'] == 'QFL'].copy()
    skew_mapping = {
        'client_1': '0.90 (Extrem)', 'client_4': '0.90 (Extrem)',
        'client_2': '0.60 (Mild)', 'client_3': '0.60 (Mild)'
    }
    df_qfl['LDS_Skew'] = df_qfl['Client'].map(skew_mapping)
    pivot = df_qfl.pivot_table(index='LDS_Skew', columns='Noise_P', values='Testgenauigkeit', aggfunc='mean')
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Robustheits-Matrix: LDS Skew vs. Rauschen (QFL)")
    plt.xlabel("Rauschwahrscheinlichkeit p")
    plt.ylabel("LDS Skew Stärke")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Master_Skew_Noise_Heatmap.png"), dpi=300)
    plt.close()

# 4. MASTER-ANALYSE ÜBER ALLE RAUSCHLEVEL

def plot_master_noise_sensitivity(df_master, save_dir):
    """Absturzkurven für QFL und Baseline über ansteigendes Rauschen."""
    setup_german_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metrics = [
        ('ROC_AUC', 'Trennschärfe (ROC-AUC)'),
        ('PR_AUC', 'Minderheiten-Resilienz (PR-AUC)')
    ]
    for ax, (m, t) in zip(axes, metrics):
        df_plot = df_master[df_master['Szenario'] != 'Zentrale Baseline']
        sns.lineplot(
            data=df_plot, x='Noise_P', y=m, hue='Noise_Type',
            style='Szenario', palette=NOISE_COLOR_MAP, marker='o', lw=2, ax=ax
        )
        ax.set_title(t)
        ax.set_xlabel("Rauschwahrscheinlichkeit p")
        format_y_axis(ax)
    plt.suptitle("Absturzkurven: Performance-Vergleich unter Rauschen", fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "Master_Noise_Sensitivitaet.png"), dpi=300)
    plt.close()

def plot_noise_type_impact_ranking(df_master, save_dir):
    #Ranking der Rauschtypen basierend auf dem Genauigkeitsverlust
    setup_german_plot_style()
    ref = df_master[df_master['Noise_P'] == 0]
    baseline_perf = ref.groupby('Szenario')['Testgenauigkeit'].mean()
    impact_data = []
    noise_types = [nt for nt in NOISE_COLOR_MAP.keys() if nt != "None"]
    for scenario in ['QFL', 'Lokale Baseline']:
        for nt in noise_types:
            df_noise = df_master[(df_master['Szenario'] == scenario) & (df_master['Noise_Type'] == nt)]
            if not df_noise.empty:
                avg_perf = df_noise[df_noise['Noise_P'] > 0]['Testgenauigkeit'].mean()
                loss = baseline_perf[scenario] - avg_perf
                impact_data.append({
                    'Szenario': scenario, 'Rauschtyp': nt, 'Genauigkeitsverlust': loss
                })
    df_impact = pd.DataFrame(impact_data)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_impact, x='Rauschtyp', y='Genauigkeitsverlust',
        hue='Szenario', palette=GLOBAL_PALETTE
    )
    ax.set_title("Ranking der Rauschtypen nach Leistungsverlust", fontweight='bold')
    ax.set_ylabel("Durchschnittlicher Verlust der Testgenauigkeit")
    ax.set_xlabel("Art des Quantenrauschens")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ranking_Rauschschaden.png"), dpi=300)
    plt.close()

def plot_critical_thresholds_annotated(df_master, save_dir):
    #Absturzkurven mit Markierung der kritischen 10%-Verlustmarke
    setup_german_plot_style()
    plt.figure(figsize=(12, 7))
    df_qfl = df_master[df_master['Szenario'] == 'QFL']
    df_p0 = df_qfl[df_qfl['Noise_P'] == 0]
    if df_p0.empty:
        return
    p0_val = df_p0['Testgenauigkeit'].mean()
    limit = p0_val * 0.90
    noise_types = [nt for nt in NOISE_COLOR_MAP.keys() if nt != "None"]
    for nt in noise_types:
        df_nt = df_qfl[df_qfl['Noise_Type'] == nt]
        stats = df_nt.groupby('Noise_P')['Testgenauigkeit'].mean()
        plt.plot(stats.index, stats.values, marker='o', label=nt, linewidth=2)
        critical_p = stats[stats < limit].index.min()
        if not pd.isna(critical_p):
            plt.axvline(x=critical_p, linestyle='--', alpha=0.3, color='gray')
    plt.axhline(y=limit, color='red', linestyle=':', alpha=0.6, label='Kritische Grenze (-10%)')
    plt.title("Analyse der kritischen Rauschgrenzen für den QFL-Leistungseinbruch", fontweight='bold')
    plt.xlabel("Rauschwahrscheinlichkeit p")
    plt.ylabel("Testgenauigkeit (QFL)")
    plt.legend(frameon=True, loc='best')
    format_y_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Kritische_Rauschschwellen.png"), dpi=300)
    plt.close()

def plot_stability_analysis(df_master, save_dir):
    #Analysiert die Standardabweichung
    setup_german_plot_style()
    stability_df = df_master.groupby(['Szenario', 'Noise_Type', 'Noise_P'])['Balancierte_Genauigkeit'].std().reset_index()
    stability_df.rename(columns={'Balancierte_Genauigkeit': 'Std_Dev'}, inplace=True)
    stability_df = stability_df[stability_df['Szenario'].isin(['Lokale Baseline', 'QFL'])]
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=stability_df, x='Noise_P', y='Std_Dev', hue='Szenario',
        style='Noise_Type', palette=GLOBAL_PALETTE, marker='o', lw=2, markersize=8
    )
    plt.title("Stabilitäts-Analyse: Varianz der Ergebnisse unter Rauschen", fontweight='bold')
    plt.xlabel("Rauschwahrscheinlichkeit p")
    plt.ylabel(r"Standardabweichung $\sigma$")
    plt.legend(frameon=True, title="Szenario / Rauschtyp", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Master_Stabilitat_Varianz.png"), dpi=300)
    plt.close()

def plot_noise_evolution_convergence(evolution_data, noise_type, save_dir):
    #Vergleicht die Konvergenzverzögerung über p
    setup_german_plot_style()
    plt.figure(figsize=(10, 6))
    p_colors = {0.0: "black", 0.01: "#56B4E9", 0.05: "#E69F00", 0.1: "#D55E00"}
    for p_val, history in evolution_data.items():
        if history is not None:
            x_axis = np.arange(1, len(history['acc']) + 1) * 3
            label = f"p = {p_val} (Referenz)" if p_val == 0 else f"p = {p_val}"
            plt.plot(
                x_axis, history['acc'], label=label,
                color=p_colors.get(p_val, "gray"), lw=2,
                marker='o' if p_val > 0 else None, markersize=4
            )
    plt.title(f"Konvergenz-Verzögerung unter {noise_type}-Rauschen", fontweight='bold')
    plt.xlabel("Epochen")
    plt.ylabel("Globale Validierungsgenauigkeit")
    plt.legend(frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    format_y_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Evolution_Convergence_{noise_type}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix_comparison(cm_baseline, cm_qfl, sub_id, save_dir):
    #Qualitativer Vergleich der Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = ['Gesund', 'Pneumonie']
    for ax, cm, title in zip([ax1, ax2], [cm_baseline, cm_qfl], ['Lokale Baseline (Noisy)', 'QFL (Noisy)']):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax, cbar=False
        )
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Vorhergesagt")
        ax.set_ylabel("Tatsächlich")
        for _, spine in ax.spines.items():
            spine.set_visible(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"CM_Vergleich_{sub_id}.png"), dpi=300)
    plt.close()