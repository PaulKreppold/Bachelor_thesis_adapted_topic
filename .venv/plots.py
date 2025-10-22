import numpy as np
import matplotlib.pyplot as plt


def plot_qfl_vs_baseline_accuracy(qfl_accuracies, baseline_accuracies, clients):

    num_clients = len(clients)
    nrows = int(np.ceil(num_clients / 2))
    ncols = 2
    if num_clients == 1:
        nrows, ncols = 1, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 18), squeeze=False)
    axes = axes.flatten()

    for idx, client_id in enumerate(clients):
        ax = axes[idx]

        qfl_curves = []
        for seed in qfl_accuracies:
            qfl_curves.append(qfl_accuracies[seed][client_id])

        min_len_qfl = min(len(c) for c in qfl_curves)
        qfl_curves = np.array([c[:min_len_qfl] for c in qfl_curves])

        mean_qfl = np.mean(qfl_curves, axis=0)
        std_qfl = np.std(qfl_curves, axis=0)
        epochs_qfl = np.arange(len(mean_qfl))

        ax.plot(epochs_qfl, mean_qfl, label="QFL Training", color="dodgerblue")
        ax.fill_between(epochs_qfl, mean_qfl - std_qfl, mean_qfl + std_qfl,
                        alpha=0.2, color="dodgerblue")

        baseline_curves = []
        for seed in baseline_accuracies:
            baseline_curves.append(baseline_accuracies[seed][client_id])

        min_len_baseline = min(len(c) for c in baseline_curves)
        baseline_curves = np.array([c[:min_len_baseline] for c in baseline_curves])

        mean_baseline = np.mean(baseline_curves, axis=0)
        std_baseline = np.std(baseline_curves, axis=0)
        epochs_baseline = np.arange(len(mean_baseline))

        ax.plot(epochs_baseline, mean_baseline, label="Baseline Training", color="darkorange")
        ax.fill_between(epochs_baseline, mean_baseline - std_baseline, mean_baseline + std_baseline,
                        alpha=0.2, color="darkorange")

        ax.set_title(f"Vergleich für Client {client_id}")
        ax.set_xlabel("Epochen")
        ax.set_ylabel("Trainings-Genauigkeit (%)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.set_ylim(50, 95)

    for i in range(num_clients, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("QFL vs. Baseline: Trainingsgenauigkeit pro Client", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_per_client_boxplots(qfl_client_results, baseline_client_results, clients):

    num_clients = len(clients)
    nrows = int(np.ceil(num_clients / 2))
    ncols = 2 if num_clients > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = np.array(axes).flatten() if num_clients > 1 else [axes]

    box_linewidth = 1.2

    for idx, client_id in enumerate(clients):
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
        ax.set_title(f'Client {client_id}: QFL vs. Baseline')
        ax.yaxis.grid(True, linestyle='--', alpha=0.25)

    for i in range(num_clients, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle('Vergleich pro Client: Finale Testgenauigkeit QFL vs. Baseline', fontsize=16)
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


