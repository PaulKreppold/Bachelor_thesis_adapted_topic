import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    plt.style.use('seaborn-v0_8-paper')  # Oder 'default'
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def plot_training_curves(histories, title, filename):
    """
    histories: Dict mit Key 'Label' und Value 'Liste von Arrays (einer pro Seed)'
    """
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for i, (label, seeds_data) in enumerate(histories.items()):
        # seeds_data ist (num_seeds, num_epochs)
        acc_mean = np.mean([s['acc'] for s in seeds_data], axis=0)
        acc_std = np.std([s['acc'] for s in seeds_data], axis=0)
        loss_mean = np.mean([s['loss'] for s in seeds_data], axis=0)
        loss_std = np.std([s['loss'] for s in seeds_data], axis=0)

        epochs = np.arange(1, len(acc_mean) + 1)

        # Accuracy Plot
        ax1.plot(epochs, acc_mean, label=label, color=colors[i])
        ax1.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.1, color=colors[i])

        # Loss Plot
        ax2.plot(epochs, loss_mean, label=label, color=colors[i])
        ax2.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.1, color=colors[i])

    ax1.set_title("Trainingsgenauigkeit")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.set_title("Trainingsverlust (BCE)")
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_academic_boxplot(data_dict, title, filename):
    """
    data_dict: {'Modellname': [Werte_Seed1, Werte_Seed2, ...]}
    """
    setup_plot_style()
    plt.figure(figsize=(8, 6))

    # DataFrame für Seaborn
    df = pd.DataFrame([(k, v) for k, l in data_dict.items() for v in l], columns=['Modell', 'Accuracy'])

    # Boxplot ohne Füllung (patch_artist=False oder fill=False)
    ax = sns.boxplot(x='Modell', y='Accuracy', data=df,
                     color='black',
                     fill=False,  # Erzeugt den "leeren" Look
                     width=0.5,
                     linewidth=1.2)

    # Einzelne Datenpunkte (Seeds) einzeichnen
    sns.stripplot(x='Modell', y='Accuracy', data=df, color='red', size=6, alpha=0.7)

    plt.title(title)
    plt.ylabel("Finale Trainingsgenauigkeit")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()