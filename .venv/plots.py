import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def setup_german_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'font.family': 'sans-serif'})


def plot_ablation_training_curves_by_layer(all_histories, save_dir):
    """Erstellt separate Plots für 2, 4 und 6 Layer."""
    setup_german_plot_style()
    layers = [2, 4, 6]

    for l in layers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        # Filter: Nur Konfigurationen mit l Layern
        l_configs = {k: v for k, v in all_histories.items() if k.startswith(f"L{l}_")}

        colors = sns.color_palette("viridis", len(l_configs))
        for (label, h_list), color in zip(l_configs.items(), colors):
            t_acc = np.mean([h['train_acc'] for h in h_list], axis=0)
            t_loss = np.mean([h['train_loss'] for h in h_list], axis=0)
            epochs = np.arange(1, len(t_acc) + 1)

            ax1.plot(epochs, t_acc, label=label.replace(f"L{l}_", ""), color=color, lw=2)
            ax2.plot(epochs, t_loss, label=label.replace(f"L{l}_", ""), color=color, lw=2)

        ax1.set_title(f"Trainingsgenauigkeit ({l} Layer)", fontweight='bold')
        ax2.set_title(f"Trainingsverlust ({l} Layer)", fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax2.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Ablation_L{l}_Curves.png"), dpi=300)
        plt.close()


def plot_top_3_configs(all_histories, df_master, save_dir):
    """Identifiziert die 3 besten Konfigurationen über alle Seeds und plottet diese."""
    setup_german_plot_style()

    # 1. Beste 3 Labels finden (nach Durchschnitts-Accuracy)
    top_3_labels = df_master.groupby("Label")["Testgenauigkeit"].mean().nlargest(3).index.tolist()

    plt.figure(figsize=(10, 6))
    colors = ["#0072B2", "#D55E00", "#009E73"]  # Blau, Orange, Grün (Wissenschafts-Farben)

    for label, color in zip(top_3_labels, colors):
        h_list = all_histories[label]
        t_acc = np.mean([h['train_acc'] for h in h_list], axis=0)
        epochs = np.arange(1, len(t_acc) + 1)
        plt.plot(epochs, t_acc, label=f"TOP {top_3_labels.index(label) + 1}: {label}", color=color, lw=3)

    plt.title("Lernkurven der insgesamt 3 besten Konfigurationen", fontweight='bold')
    plt.xlabel("Epochen")
    plt.ylabel("Trainingsgenauigkeit")
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ablation_Top_3_Comparison.png"), dpi=300)
    plt.close()


def plot_test_acc_boxplot(df, save_dir):
    setup_german_plot_style()
    plt.figure(figsize=(15, 7))
    # Boxplot nach Layer gruppiert
    ax = sns.boxplot(data=df, x='Label', y='Testgenauigkeit', hue='Encoding', palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title("Vergleich der Testgenauigkeit über alle 18 Konfigurationen", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ablation_Overall_Boxplot.png"), dpi=300)
    plt.close()