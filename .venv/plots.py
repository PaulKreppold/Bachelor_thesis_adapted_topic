"""
Comprehensive Visualization Module for Quantum Federated Learning Analysis
===========================================================================

Provides three main visualization suites:
1. LDS Bias Analysis (Motivation for QFL)
2. Isolated Noise Study (Core experimental results)
3. Training Dynamics & Convergence Analysis

Author: [Your Name]
Date: 2026-01-28
Version: 2.0 (Fixed noise handling)
"""

import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SCENARIO_COLORS = {
    "Lokale Baseline": "#E15759",  # Rot
    "QFL": "#4E79A7",              # Dunkelblau
    "QFL_Global": "#4E79A7",       # Dunkelblau (Synonym für Master-Plots)
    "QFL_Local": "#76B7B2",        # Hellblau/Teal (Personalisiert)
    "Zentrale Baseline": "#59A14F" # Grün
}

NOISE_COLORS = {
    "bitflip": "#F28E2B",  # Orange
    "phaseflip": "#76B7B2",  # Teal
    "depolarizing": "#B07AA1",  # Purple
    "None": "#4A4A4A"  # Dark Gray
}

LABEL_MAP = {
    'client_1': 'PneumoniaMNIST',
    'client_2': 'RSNA (60/40)',
    'client_3': 'RSNA (40/60)',
    'client_4': 'CheXpert (25/75)'
}

# IEEE-compliant figure sizes (in inches)
FIGSIZE_SINGLE = (3.5, 3.0)
FIGSIZE_DOUBLE = (7.2, 3.0)
FIGSIZE_TALL = (5.5, 4.0)


# =============================================================================
# STYLING UTILITIES
# =============================================================================

def get_paper_style_context():
    """Returns a context manager for publication-quality plots."""
    return plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#2B2B2B',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.axisbelow': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': '#CCCCCC',
        'legend.frameon': True,
    })


def format_y_axis(ax, precision=2, percentage=False):
    """Formats Y-axis with fixed decimal places."""
    if percentage:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.{precision}%}'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{precision}f'))


def get_noise_label(noise_type, p):
    """Creates descriptive labels for noise scenarios."""
    if str(noise_type) == "None" or p == 0:
        return "Noiseless"
    noise_names = {"bitflip": "Bit-Flip", "phaseflip": "Phase-Flip", "depolarizing": "Depolarizing"}
    return f"{noise_names.get(noise_type, noise_type)} (p={p})"


def apply_transparent_boxes(ax, alpha=0.3):
    """Makes boxplot boxes transparent with colored edges."""
    for patch in ax.patches:
        color = patch.get_facecolor()
        patch.set_facecolor((color[0], color[1], color[2], alpha))
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)


# =============================================================================
# SECTION 1: LDS BIAS ANALYSIS & MOTIVATION
# =============================================================================

def plot_lds_performance_scatter(df_master, save_path):
    """Scatter-Plot: Performance vs. Skew Severity."""
    with get_paper_style_context():
        df_plot = df_master[df_master['Noise_P'] == 0].copy()
        if df_plot.empty:
            print(f"⚠️  Skipping {save_path}: No noiseless data available")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        scenarios = ['Lokale Baseline', 'QFL']
        markers = {'Lokale Baseline': 'x', 'QFL': 'o'}

        for scen in scenarios:
            data = df_plot[df_plot['Szenario'] == scen]
            if data.empty: continue
            sns.regplot(data=data, x='LDS_Ratio', y='Testgenauigkeit',
                        label=scen, marker=markers.get(scen, 'o'), scatter_kws={'alpha': 0.5},
                        color=SCENARIO_COLORS.get(scen), ax=ax)

        ax.set_xlabel('Skew Severity (Majority Class Ratio)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Robustness Against Label Distribution Skew', fontweight='bold')

        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_lds_bias_reduction_barplot(df, save_path):
    """Shows QFL improves minority class recall on extreme LDS clients."""
    with get_paper_style_context():
        lds_targets = [('client_1', '1'), ('client_4', '1')]
        df_lds = df[df.apply(lambda x: (str(x['Client']), str(x['Subclient'])) in lds_targets, axis=1)].copy()

        if df_lds.empty:
            print(f"⚠️  Skipping {save_path}: No LDS client data available")
            return

        df_lds['Client_Label'] = df_lds['Client'].map({
            'client_1': 'PneumoniaMNIST\n(10/90 N/P)',
            'client_4': 'CheXpert\n(90/10 N/P)'
        })

        fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
        x = np.arange(2)
        width = 0.35
        scenarios = ['Lokale Baseline', 'QFL']
        client_order = ['PneumoniaMNIST\n(10/90 N/P)', 'CheXpert\n(90/10 N/P)']

        for i, scenario in enumerate(scenarios):
            df_scenario = df_lds[df_lds['Szenario'] == scenario]
            if df_scenario.empty:
                continue

            means = df_scenario.groupby('Client_Label')['Recall_Minderheit'].mean().reindex(client_order)
            stds = df_scenario.groupby('Client_Label')['Recall_Minderheit'].std().reindex(client_order)

            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, means, width, label=scenario,
                          color=SCENARIO_COLORS[scenario], alpha=0.85, edgecolor='black')
            ax.errorbar(x + offset, means, yerr=stds, fmt='none', ecolor='black', capsize=3)

            for bar, mean_val in zip(bars, means):
                if not pd.isna(mean_val):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel('Minority Class Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(client_order)
        ax.set_ylim([0, 1.0])
        ax.legend(title='Approach', loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_lds_advantage_erosion_under_noise(df_master, save_path):
    """Shows how QFL's advantage on LDS clients erodes under noise."""
    with get_paper_style_context():
        # Filter for noise scenarios only
        df_noisy = df_master[(df_master['Noise_P'] > 0) & (df_master['Noise_Type'] != 'None')].copy()

        if df_noisy.empty:
            print(f"⚠️  Skipping {save_path}: No noise data available")
            return

        lds_targets = [('client_1', '1'), ('client_4', '1')]
        df_lds = df_noisy[df_noisy.apply(
            lambda x: (str(x['Client']), str(x['Subclient'])) in lds_targets, axis=1
        )].copy()

        if df_lds.empty:
            print(f"⚠️  Skipping {save_path}: No LDS client data available")
            return

        qfl = df_lds[df_lds['Szenario'] == 'QFL']
        local = df_lds[df_lds['Szenario'] == 'Lokale Baseline']

        if qfl.empty or local.empty:
            print(f"⚠️  Skipping {save_path}: Missing QFL or Local data")
            return

        merged = pd.merge(
            qfl, local,
            on=['Seed', 'Noise_Type', 'Noise_P', 'Client'],
            suffixes=('_qfl', '_local')
        )

        if merged.empty:
            print(f"⚠️  Skipping {save_path}: No matching data")
            return

        merged['Recall_Advantage'] = merged['Recall_Minderheit_qfl'] - merged['Recall_Minderheit_local']

        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        has_data = False

        for noise_type in ['bitflip', 'phaseflip', 'depolarizing']:
            df_noise = merged[merged['Noise_Type'] == noise_type]
            if df_noise.empty:
                continue

            stats = df_noise.groupby('Noise_P')['Recall_Advantage'].agg(['mean', 'std']).reset_index()
            if stats.empty:
                continue

            has_data = True
            ax.plot(stats['Noise_P'], stats['mean'],
                    label=noise_type.capitalize(),
                    color=NOISE_COLORS[noise_type],
                    linewidth=2, marker='o')
            ax.fill_between(stats['Noise_P'],
                            stats['mean']-stats['std'],
                            stats['mean']+stats['std'],
                            alpha=0.2,
                            color=NOISE_COLORS[noise_type])

        if not has_data:
            plt.close()
            print(f"⚠️  Skipping {save_path}: No plottable data")
            return

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Noise Probability (p)')
        ax.set_ylabel('Recall Advantage (QFL − Local)')
        ax.set_title('QFL Bias Mitigation Robustness', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_balanced_accuracy_comparison(df, save_path):
    """Comparison using Balanced Accuracy."""
    with get_paper_style_context():
        lds_targets = [('client_1', '1'), ('client_4', '1')]
        df_lds = df[df.apply(lambda x: (str(x['Client']), str(x['Subclient'])) in lds_targets, axis=1)].copy()

        if df_lds.empty:
            print(f"⚠️  Skipping {save_path}: No LDS client data available")
            return

        df_lds['Client_Label'] = df_lds['Client'].map({'client_1': 'PneumoniaMNIST', 'client_4': 'CheXpert'})

        fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
        sns.boxplot(data=df_lds, x='Client_Label', y='Balancierte_Genauigkeit',
                    hue='Szenario', palette=SCENARIO_COLORS, width=0.6, ax=ax, showmeans=True)
        apply_transparent_boxes(ax)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_xlabel('')
        ax.set_ylim([0.5, 1.0])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_pr_auc_lds_clients(df, save_path):
    """PR-AUC focus on minority class."""
    with get_paper_style_context():
        lds_targets = [('client_1', '1'), ('client_4', '1')]
        df_lds = df[df.apply(lambda x: (str(x['Client']), str(x['Subclient'])) in lds_targets, axis=1)].copy()

        if df_lds.empty:
            print(f"⚠️  Skipping {save_path}: No LDS client data available")
            return

        fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
        sns.violinplot(data=df_lds, x='Client', y='PR_AUC_Minority',
                       hue='Szenario', palette=SCENARIO_COLORS, ax=ax)
        ax.set_ylabel('PR-AUC (Minority Focus)')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_lds_noise_interaction_heatmap(df_master, save_path, mode='detailed'):
    """Heatmap showing interaction between LDS severity and noise."""
    with get_paper_style_context():
        # Filter out noiseless scenarios (Noise_P = 0 or Noise_Type = 'None')
        df_noisy = df_master[(df_master['Noise_P'] > 0) & (df_master['Noise_Type'] != 'None')].copy()

        if df_noisy.empty:
            print(f"⚠️  Skipping {save_path}: No noise data available (likely noiseless scenario)")
            return

        def categorize(row):
            c, s = row['Client'], str(row['Subclient'])
            if mode == 'simple':
                if (c == 'client_1' and s == '1') or (c == 'client_4' and s == '1'):
                    return 'Extreme (≤10%)'
                return 'Moderate (25-75%)' if c in ['client_1', 'client_4'] else 'Mild (40-60%)'
            else:
                mapping = {
                    ('client_1','1'): 'Extr. (10/90)',
                    ('client_4','1'): 'Extr. (90/10)',
                    ('client_2','1'): 'Mild (60/40)',
                    ('client_3','1'): 'Mild (40/60)'
                }
                return mapping.get((c, s), 'Moderate')

        df_noisy['LDS_Cat'] = df_noisy.apply(categorize, axis=1)

        qfl = df_noisy[df_noisy['Szenario'] == 'QFL']
        local = df_noisy[df_noisy['Szenario'] == 'Lokale Baseline']

        if qfl.empty or local.empty:
            print(f"⚠️  Skipping {save_path}: Insufficient data for comparison")
            return

        merged = pd.merge(
            qfl, local,
            on=['Seed', 'Noise_Type', 'Noise_P', 'Client', 'Subclient', 'LDS_Cat'],
            suffixes=('_qfl', '_local')
        )

        if merged.empty:
            print(f"⚠️  Skipping {save_path}: No matching data after merge")
            return

        merged['Advantage'] = merged['Balancierte_Genauigkeit_qfl'] - merged['Balancierte_Genauigkeit_local']

        pivot = merged.pivot_table(
            index='LDS_Cat',
            columns='Noise_P',
            values='Advantage',
            aggfunc='mean'
        )

        if pivot.empty:
            print(f"⚠️  Skipping {save_path}: Empty pivot table")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('LDS × Noise Interaction (BA Advantage)', fontweight='bold')
        ax.set_xlabel('Noise Probability (p)')
        ax.set_ylabel('LDS Category')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


# Alias functions for the orchestrator
def plot_lds_noise_interaction_heatmap_simplified(df, path):
    plot_lds_noise_interaction_heatmap(df, path, 'simple')


def plot_lds_noise_interaction_heatmap_extreme_only(df, path):
    plot_lds_noise_interaction_heatmap(df, path, 'detailed')


# =============================================================================
# SECTION 2: TRAINING DYNAMICS
# =============================================================================

def plot_training_comparison(histories_baseline, histories_qfl, title, save_path):
    """Panel plot: Train Accuracy & Loss."""
    with get_paper_style_context():
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        for i, (key, ylabel) in enumerate([('acc', 'Accuracy'), ('loss', 'Loss')]):
            ax = axes[i]
            for data, lab, col, ls in [(histories_qfl, 'QFL', SCENARIO_COLORS['QFL'], '-'),
                                       (histories_baseline, 'Local', SCENARIO_COLORS['Lokale Baseline'], '--')]:
                if not data: continue
                curves = [h[f'train_{key}'] for h in data]
                if not curves: continue
                mean = np.mean(curves, axis=0)
                ax.plot(range(1, len(mean)+1), mean, label=lab, color=col, linestyle=ls)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Epoch')
            ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_convergence_speed_comparison(histories_baseline, histories_qfl, save_path):
    """Epochs to 95% final accuracy."""
    with get_paper_style_context():
        data = []
        for h_list, scen in [(histories_qfl, 'QFL'), (histories_baseline, 'Lokale Baseline')]:
            for h in h_list:
                curve = h.get('train_acc', [])
                if not curve: continue
                target = 0.95 * curve[-1]
                epoch = next((i+1 for i, v in enumerate(curve) if v >= target), len(curve))
                data.append({'Scenario': scen, 'Epochs': epoch})

        if not data:
            print(f"⚠️  Skipping {save_path}: No training data available")
            return

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        sns.boxplot(data=df, x='Scenario', y='Epochs', hue='Scenario', palette=SCENARIO_COLORS, ax=ax, legend=False)
        ax.set_title('Convergence Speed')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


# =============================================================================
# SECTION 3: ISOLATED NOISE STUDY
# =============================================================================

def plot_global_comparison_boxplot(df, save_path):
    """Global Test Accuracy comparison."""
    with get_paper_style_context():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.boxplot(data=df, x='Szenario', y='Testgenauigkeit', hue='Szenario',
                    order=["Zentrale Baseline", "Lokale Baseline", "QFL"],
                    palette=SCENARIO_COLORS, ax=ax, legend=False, showmeans=True)
        apply_transparent_boxes(ax)
        format_y_axis(ax)
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_domain_specific_boxplot(df, save_path):
    """Accuracy broken down by clinical domains."""
    with get_paper_style_context():
        df_plot = df.copy()
        df_plot['Domain'] = df_plot['Client'].map(LABEL_MAP)
        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
        sns.boxplot(data=df_plot, x='Domain', y='Testgenauigkeit', hue='Szenario', palette=SCENARIO_COLORS, ax=ax)
        apply_transparent_boxes(ax)
        ax.legend(loc='lower right')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_noise_sensitivity_curves(df_master, save_path):
    """Degradation over noise levels (ROC-AUC & PR-AUC)."""
    with get_paper_style_context():
        # Filter for noise scenarios
        df_noisy = df_master[(df_master['Noise_P'] > 0) & (df_master['Noise_Type'] != 'None')].copy()

        if df_noisy.empty:
            print(f"⚠️  Skipping {save_path}: No noise data available")
            return

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
        metrics = [('ROC_AUC_Global', 'ROC-AUC'), ('PR_AUC_Minority', 'PR-AUC')]

        has_data = False

        for i, (met, ylabel) in enumerate(metrics):
            ax = axes[i]

            for scen in ['QFL', 'Lokale Baseline']:
                for nt in ['bitflip', 'phaseflip', 'depolarizing']:
                    sub = df_noisy[(df_noisy['Szenario'] == scen) & (df_noisy['Noise_Type'] == nt)]
                    if sub.empty:
                        continue

                    stats = sub.groupby('Noise_P')[met].mean()
                    if stats.empty:
                        continue

                    has_data = True
                    ax.plot(stats.index, stats.values,
                            label=f"{nt} ({scen})",
                            color=NOISE_COLORS[nt],
                            linestyle='-' if scen=='QFL' else '--',
                            marker='o', markersize=3)

            ax.set_ylabel(ylabel)
            ax.set_xlabel('p')

        if not has_data:
            plt.close()
            print(f"⚠️  Skipping {save_path}: No plottable data")
            return

        axes[1].legend(fontsize=6, bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_noise_impact_ranking(df_master, save_path):
    """Bar chart: Accuracy loss per noise type."""
    with get_paper_style_context():
        df_noiseless = df_master[df_master['Noise_P'] == 0]
        df_noisy = df_master[(df_master['Noise_P'] > 0) & (df_master['Noise_Type'] != 'None')]

        if df_noiseless.empty or df_noisy.empty:
            print(f"⚠️  Skipping {save_path}: Need both noiseless and noisy data")
            return

        ref = df_noiseless.groupby('Szenario')['Testgenauigkeit'].mean()
        data = []

        for scen in ['QFL', 'Lokale Baseline']:
            for nt in ['bitflip', 'phaseflip', 'depolarizing']:
                perf = df_noisy[
                    (df_noisy['Szenario'] == scen) &
                    (df_noisy['Noise_Type'] == nt)
                ]['Testgenauigkeit'].mean()

                if not pd.isna(perf) and scen in ref:
                    data.append({
                        'Scenario': scen,
                        'Noise': nt,
                        'Loss': ref[scen] - perf
                    })

        if not data:
            print(f"⚠️  Skipping {save_path}: No comparison data available")
            return

        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=pd.DataFrame(data), x='Noise', y='Loss',
                    hue='Scenario', palette=SCENARIO_COLORS, ax=ax)
        ax.set_title('Average Accuracy Loss')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_critical_thresholds(df_master, save_path):
    """Identifies noise levels where QFL drops >10%."""
    with get_paper_style_context():
        df_qfl = df_master[df_master['Szenario'] == 'QFL']

        df_qfl_noiseless = df_qfl[df_qfl['Noise_P'] == 0]
        df_qfl_noisy = df_qfl[(df_qfl['Noise_P'] > 0) & (df_qfl['Noise_Type'] != 'None')]

        if df_qfl_noiseless.empty or df_qfl_noisy.empty:
            print(f"⚠️  Skipping {save_path}: Need both noiseless and noisy QFL data")
            return

        p0_perf = df_qfl_noiseless['Testgenauigkeit'].mean()
        threshold = p0_perf * 0.90

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        has_data = False

        for nt in ['bitflip', 'phaseflip', 'depolarizing']:
            stats = df_qfl_noisy[df_qfl_noisy['Noise_Type'] == nt].groupby('Noise_P')['Testgenauigkeit'].mean()
            if stats.empty:
                continue
            has_data = True
            ax.plot(stats.index, stats.values, marker='o', label=nt, color=NOISE_COLORS[nt])

        if not has_data:
            plt.close()
            print(f"⚠️  Skipping {save_path}: No plottable data")
            return

        ax.axhline(y=threshold, color='red', linestyle='--', label='10% Drop')
        ax.set_xlabel('Noise Probability (p)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Critical Noise Thresholds', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


def plot_stability_analysis(df_master, save_path):
    """Variance of results over noise."""
    with get_paper_style_context():
        df_noisy = df_master[(df_master['Noise_P'] > 0) & (df_master['Noise_Type'] != 'None')]

        if df_noisy.empty:
            print(f"⚠️  Skipping {save_path}: No noise data available")
            return

        stats = df_noisy.groupby(['Szenario', 'Noise_Type', 'Noise_P'])['Testgenauigkeit'].std().reset_index()
        stats_filtered = stats[stats['Szenario'].isin(['QFL', 'Lokale Baseline'])]

        if stats_filtered.empty:
            print(f"⚠️  Skipping {save_path}: No variance data available")
            return

        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.lineplot(data=stats_filtered,
                     x='Noise_P', y='Testgenauigkeit',
                     hue='Noise_Type', style='Szenario',
                     palette=NOISE_COLORS, ax=ax)
        ax.set_ylabel('Std Dev (σ)')
        ax.set_xlabel('Noise Probability (p)')
        ax.set_title('Performance Stability Under Noise', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


# =============================================================================
# SECTION 4: CONFUSION MATRICES
# =============================================================================

def plot_lds_confusion_matrix_comparison(cm_local, cm_qfl, client_id, noise_label, save_path):
    """Visualizes debiasing through CMs."""
    with get_paper_style_context():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for ax, cm, title in zip(axes, [cm_local, cm_qfl], ['Local (Biased)', 'QFL (Debiased)']):
            norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(norm, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax, cbar=False)
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        plt.suptitle(f'{client_id} - {noise_label}', fontsize=10, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Saved: {save_path}")


# =============================================================================
# ORCHESTRATION
# =============================================================================

def generate_scenario_plots(df, scenario_label, save_dir):
    """Generates all plots for a single scenario (e.g., Noiseless, bitflip_p0.1)."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 Generating scenario plots for: {scenario_label}")

    plot_global_comparison_boxplot(df, os.path.join(save_dir, "01_Global.png"))
    plot_domain_specific_boxplot(df, os.path.join(save_dir, "02_Domain.png"))
    plot_lds_bias_reduction_barplot(df, os.path.join(save_dir, "03_Bias_Recall.png"))
    plot_balanced_accuracy_comparison(df, os.path.join(save_dir, "04_BA.png"))
    plot_pr_auc_lds_clients(df, os.path.join(save_dir, "05_PR_AUC.png"))

    print(f"✅ Scenario plots complete for: {scenario_label}\n")


def generate_master_plots(df_master, save_dir):
    """Generates master plots with graceful handling of missing noise data."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 Generating master plots across all scenarios")

    # Always generate (works with noiseless data)
    plot_lds_performance_scatter(df_master, os.path.join(save_dir, "Master_00_LDS_Scatter.png"))

    # Only generate if noise data exists
    plot_noise_sensitivity_curves(df_master, os.path.join(save_dir, "Master_01_Sensitivity.png"))
    plot_noise_impact_ranking(df_master, os.path.join(save_dir, "Master_02_Ranking.png"))
    plot_critical_thresholds(df_master, os.path.join(save_dir, "Master_03_Thresholds.png"))
    plot_stability_analysis(df_master, os.path.join(save_dir, "Master_04_Stability.png"))
    plot_lds_advantage_erosion_under_noise(df_master, os.path.join(save_dir, "Master_05_Erosion.png"))
    plot_lds_noise_interaction_heatmap(df_master, os.path.join(save_dir, "Master_06a_Heatmap_Det.png"))
    plot_lds_noise_interaction_heatmap_simplified(df_master, os.path.join(save_dir, "Master_06b_Heatmap_Simp.png"))

    print(f"✅ Master plots complete\n")


def generate_training_plots(h_local, h_qfl, label, save_dir):
    """Generates training dynamics plots."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 Generating training plots for: {label}")

    plot_training_comparison(h_local, h_qfl, f"Training {label}", os.path.join(save_dir, "Train_Dynamics.png"))
    plot_convergence_speed_comparison(h_local, h_qfl, os.path.join(save_dir, "Convergence.png"))

    print(f"✅ Training plots complete for: {label}\n")


# =============================================================================
# EXPORT SUMMARY
# =============================================================================

__all__ = [
    # LDS Bias Analysis
    'plot_lds_bias_reduction_barplot',
    'plot_lds_advantage_erosion_under_noise',
    'plot_balanced_accuracy_comparison',
    'plot_pr_auc_lds_clients',
    'plot_lds_performance_scatter',
    'plot_lds_noise_interaction_heatmap',
    'plot_lds_noise_interaction_heatmap_simplified',
    'plot_lds_noise_interaction_heatmap_extreme_only',

    # Training Dynamics
    'plot_training_comparison',
    'plot_convergence_speed_comparison',

    # Isolated Noise Study
    'plot_global_comparison_boxplot',
    'plot_domain_specific_boxplot',
    'plot_noise_sensitivity_curves',
    'plot_noise_impact_ranking',
    'plot_critical_thresholds',
    'plot_stability_analysis',

    # Confusion Matrices
    'plot_lds_confusion_matrix_comparison',

    # Orchestration
    'generate_scenario_plots',
    'generate_master_plots',
    'generate_training_plots',
]