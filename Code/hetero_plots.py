import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import json
import glob
import os


def plot_qfl_noise_comparison(scenario_dirs, labels, colors, save_path, suptitle):

    #Liest die json-Dateien der übergebenen Ordner ein, mittelt über Clients/Seeds
    #und erstellt den Plot.

    epochs = np.arange(0, 15)
    plt.figure(figsize=(12, 5.5))

    all_scenarios_acc = []
    all_scenarios_loss = []
    data_found = False

    for s_dir in scenario_dirs:
        # Sucht alle JSON-Dateien im aktuellen Szenario-Ordner
        json_files = glob.glob(os.path.join(s_dir, '*_full_results.json'))

        if not json_files:
            print(f"  -> Warnung: Keine JSON-Dateien in {s_dir} gefunden!")

        scenario_seed_accs = []
        scenario_seed_losses = []

        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)

            client_accs = []
            client_losses = []

            if 'qfl_local_history' in data.get('history', {}):
                for client_id, metrics in data['history']['qfl_local_history'].items():
                    if 'train_acc' in metrics and 'train_loss' in metrics:
                        client_accs.append(metrics['train_acc'])
                        client_losses.append(metrics['train_loss'])

            if client_accs and client_losses:
                mean_client_acc = np.mean(client_accs, axis=0)
                mean_client_loss = np.mean(client_losses, axis=0)

                scenario_seed_accs.append(mean_client_acc)
                scenario_seed_losses.append(mean_client_loss)
                data_found = True

        all_scenarios_acc.append(scenario_seed_accs)
        all_scenarios_loss.append(scenario_seed_losses)

    if not data_found:
        print(f"Überspringe Plot '{save_path}', da keine Daten geladen werden konnten.")
        plt.close()
        return

    # Linker Plot entspricht der Trainingsgenauigkeit
    ax1 = plt.subplot(1, 2, 1)
    min_acc, max_acc = float('inf'), float('-inf')

    for i, label in enumerate(labels):
        if i >= len(all_scenarios_acc) or not all_scenarios_acc[i]:
            continue
        m = np.mean(all_scenarios_acc[i], axis=0)
        s = np.std(all_scenarios_acc[i], axis=0)
        plt.plot(epochs, m, label=label, color=colors[i], linewidth=2)
        plt.fill_between(epochs, m - s, m + s, color=colors[i], alpha=0.15)
        min_acc = min(min_acc, np.min(m - s))
        max_acc = max(max_acc, np.max(m + s))

    plt.title('QFL Trainingsgenauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    margin_x = 14 * 0.05
    plt.xlim(0 - margin_x, 14 + margin_x)
    plt.xticks(np.arange(0, 15))

    if min_acc != float('inf'):
        locator = MaxNLocator(nbins=6)
        ticks = locator.tick_values(min_acc, max_acc)
        ax1.set_yticks(ticks)
        tm = (ticks[-1] - ticks[0]) * 0.05
        ax1.set_ylim(ticks[0] - tm, ticks[-1] + tm)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()

    # rechter Plot entspricht dem Trainingsverlust
    ax2 = plt.subplot(1, 2, 2)
    min_loss, max_loss = float('inf'), float('-inf')

    for i, label in enumerate(labels):
        if i >= len(all_scenarios_loss) or not all_scenarios_loss[i]:
            continue
        m = np.mean(all_scenarios_loss[i], axis=0)
        s = np.std(all_scenarios_loss[i], axis=0)
        plt.plot(epochs, m, label=label, color=colors[i], linewidth=2)
        plt.fill_between(epochs, m - s, m + s, color=colors[i], alpha=0.15)
        min_loss = min(min_loss, np.min(m - s))
        max_loss = max(max_loss, np.max(m + s))

    plt.title('QFL Trainingsverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.xlim(0 - margin_x, 14 + margin_x)
    plt.xticks(np.arange(0, 15))

    if min_loss != float('inf'):
        locator = MaxNLocator(nbins=6)
        ticks = locator.tick_values(min_loss, max_loss)
        ax2.set_yticks(ticks)
        tm = (ticks[-1] - ticks[0]) * 0.05
        ax2.set_ylim(ticks[0] - tm, ticks[-1] + tm)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# main loop zur Code Ausführung
if __name__ == '__main__':
    base_path = '/Users/paulkreppold/Documents/Neuer_S3_Export'
    probs = ['0.01', '0.03', '0.05', '0.1']
    plot_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    labels = [f'p = {p}' for p in probs]

    # 1. BIAS GLOBAL NOISE
    bg_dirs = [os.path.join(base_path, 'BIAS_GLOBAL_NOISE', f'BIAS_GLOBAL_NOISE_p{p}') for p in probs]
    plot_qfl_noise_comparison(bg_dirs, labels, plot_colors, 'hetero_plots/01_bias_global_noise.png',
                              'Globales Rauschen (Bias-Szenario)')

    # 2. BIAS TRAP
    bt_dirs = [os.path.join(base_path, 'BIAS_TRAP', f'BIAS_TRAP_p{p}') for p in probs]
    plot_qfl_noise_comparison(bt_dirs, labels, plot_colors, 'hetero_plots/02_bias_trap.png',
                              'Bias-Trap-Szenario (Gezieltes Rauschen)')

    # BIAS REALISM
    br_dirs = [os.path.join(base_path, 'BIAS_REALISM', f'BIAS_REALISM_p{p}') for p in probs]
    plot_qfl_noise_comparison(br_dirs, labels, plot_colors, 'hetero_plots/03_bias_realism.png',
                              'Klinischer Realismus (Heterogener Rausch-Mix)')

    # OUTLIER SZENARIEN
    outlier_probs = ['0.05', '0.1']
    outlier_labels = [f'p = {p}' for p in outlier_probs]
    outlier_colors = [plot_colors[2], plot_colors[3]]  # orange & rot

    # BIAS OUTLIER DEPOLARIZING
    bod_dirs = [os.path.join(base_path, 'BIAS_OUTLIER', f'BIAS_OUTLIER_DEPOLARIZING_p{p}') for p in outlier_probs]
    plot_qfl_noise_comparison(bod_dirs, outlier_labels, outlier_colors, 'hetero_plots/04_bias_outlier_dep.png',
                              'Rausch-Ausreißer (Depolarisierung)')

    # BIAS OUTLIER AMPLITUDE DAMPING
    boa_dirs = [os.path.join(base_path, 'BIAS_OUTLIER', f'BIAS_OUTLIER_AMPLITUDE_DAMPING_p{p}') for p in outlier_probs]
    plot_qfl_noise_comparison(boa_dirs, outlier_labels, outlier_colors, 'hetero_plots/05_bias_outlier_amp.png',
                              'Rausch-Ausreißer (Amplitudendämpfung)')

