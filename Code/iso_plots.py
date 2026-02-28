import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import json
import glob
import os


def plot_qfl_noise_comparison(scenario_dirs, labels, colors, save_path, suptitle):
    """
    Liest die json-Dateien der übergebenen Ordner ein, mittelt über Clients/Seeds
    und erstellt den Plot.
    - Y-Achsen: Saubere Beschriftung, Kurve bleibt innerhalb der Zahlen, PLUS Puffer zum Rand.
    - X-Achse (Epochen): von 0 bis 14 mit 5% Puffer links und rechts.
    """
    epochs = np.arange(0, 15)
    plt.figure(figsize=(12, 5.5))

    all_scenarios_acc = []
    all_scenarios_loss = []
    data_found = False

    for s_dir in scenario_dirs:
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

    # ==========================================
    # Linker Plot: Trainingsgenauigkeit
    # ==========================================
    ax1 = plt.subplot(1, 2, 1)

    min_acc = float('inf')
    max_acc = float('-inf')

    for i, label in enumerate(labels):
        if not all_scenarios_acc[i]:
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

    # 5% Puffer links und rechts für die X-Achse
    margin_x = 14 * 0.05
    plt.xlim(0 - margin_x, 14 + margin_x)
    plt.xticks(np.arange(0, 15))

    if min_acc != float('inf') and max_acc != float('-inf'):
        # Berechne schöne Skalen-Schritte (Ticks) für den Datenbereich
        locator = MaxNLocator(nbins=6)
        ticks = locator.tick_values(min_acc, max_acc)
        # Zwinge die Achse, genau diese Ticks zu nutzen
        ax1.set_yticks(ticks)

        # Puffer ZUSÄTZLICH zu den Ticks hinzufügen, damit sie nicht am Rand kleben
        tick_margin_y = (ticks[-1] - ticks[0]) * 0.05
        ax1.set_ylim(ticks[0] - tick_margin_y, ticks[-1] + tick_margin_y)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()

    # ==========================================
    # Rechter Plot: Trainingsverlust
    # ==========================================
    ax2 = plt.subplot(1, 2, 2)

    min_loss = float('inf')
    max_loss = float('-inf')

    for i, label in enumerate(labels):
        if not all_scenarios_loss[i]:
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

    # X-Achse
    plt.xlim(0 - margin_x, 14 + margin_x)
    plt.xticks(np.arange(0, 15))

    if min_loss != float('inf') and max_loss != float('-inf'):
        # Berechne schöne Skalen-Schritte (Ticks) für den Datenbereich
        locator = MaxNLocator(nbins=6)
        ticks = locator.tick_values(min_loss, max_loss)
        ax2.set_yticks(ticks)

        # Puffer ZUSÄTZLICH zu den Ticks hinzufügen
        tick_margin_y = (ticks[-1] - ticks[0]) * 0.05
        ax2.set_ylim(ticks[0] - tick_margin_y, ticks[-1] + tick_margin_y)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()

    # ==========================================
    # Haupttitel & Speichern
    # ==========================================
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()  # Verhindert, dass 5 Plots gleichzeitig offen bleiben


# ==========================================
# Ausführung des Codes
# ==========================================
if __name__ == '__main__':
    # Basis-Pfade
    bit_flip_base = '/Users/paulkreppold/Documents/Neuer_S3_Export/ISO_BIT_FLIP'
    phase_flip_base = '/Users/paulkreppold/Documents/Neuer_S3_Export/ISO_PHASE_FLIP'
    depolarizing_base = '/Users/paulkreppold/Documents/Neuer_S3_Export/ISO_DEPOLARIZING'
    amplitude_damping_base = '/Users/paulkreppold/Documents/Neuer_S3_Export/ISO_AMPLITUDE_DAMPING'
    phase_damping_base = '/Users/paulkreppold/Documents/Neuer_S3_Export/ISO_PHASE_DAMPING'

    # Listen mit den jeweiligen p-Ordnern generieren
    bit_flip_dirs = [os.path.join(bit_flip_base, f'ISO_BIT_FLIP_p{p}') for p in ['0.01', '0.03', '0.05', '0.1']]
    phase_flip_dirs = [os.path.join(phase_flip_base, f'ISO_PHASE_FLIP_p{p}') for p in ['0.01', '0.03', '0.05', '0.1']]
    depolarizing_dirs = [os.path.join(depolarizing_base, f'ISO_DEPOLARIZING_p{p}') for p in
                         ['0.01', '0.03', '0.05', '0.1']]
    amplitude_damping_dirs = [os.path.join(amplitude_damping_base, f'ISO_AMPLITUDE_DAMPING_p{p}') for p in
                              ['0.01', '0.03', '0.05', '0.1']]
    phase_damping_dirs = [os.path.join(phase_damping_base, f'ISO_PHASE_DAMPING_p{p}') for p in
                          ['0.01', '0.03', '0.05', '0.1']]

    # Gemeinsame Parameter
    scenario_labels = ['p = 0.01', 'p = 0.03', 'p = 0.05', 'p = 0.10']
    plot_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    # 1. Bit-Flip Plot
    print("Erstelle Bit-Flip Plot...")
    plot_qfl_noise_comparison(
        scenario_dirs=bit_flip_dirs,
        labels=scenario_labels,
        colors=plot_colors,
        save_path='iso_plots/01_training_dynamics_bit_flip.png',
        suptitle='Bit-Flip-Kanal'
    )

    # 2. Phase-Flip Plot
    print("Erstelle Phase-Flip Plot...")
    plot_qfl_noise_comparison(
        scenario_dirs=phase_flip_dirs,
        labels=scenario_labels,
        colors=plot_colors,
        save_path='iso_plots/02_training_dynamics_phase_flip.png',
        suptitle='Phasen-Flip-Kanal'
    )

    # 3. Depolarizing Plot
    print("Erstelle Depolarizing Plot...")
    plot_qfl_noise_comparison(
        scenario_dirs=depolarizing_dirs,
        labels=scenario_labels,
        colors=plot_colors,
        save_path='iso_plots/03_training_dynamics_depolarizing.png',
        suptitle='Depolarisierungskanal'
    )

    # 4. Amplitude Damping Plot
    print("Erstelle Amplitude Damping Plot...")
    plot_qfl_noise_comparison(
        scenario_dirs=amplitude_damping_dirs,
        labels=scenario_labels,
        colors=plot_colors,
        save_path='iso_plots/04_training_dynamics_amplitude_damping.png',
        suptitle='Amplitudendämpfungskanal'
    )

    # 5. Phase Damping Plot
    print("Erstelle Phase Damping Plot...")
    plot_qfl_noise_comparison(
        scenario_dirs=phase_damping_dirs,
        labels=scenario_labels,
        colors=plot_colors,
        save_path='iso_plots/05_training_dynamics_phase_damping.png',
        suptitle='Phasendämpfungskanal'
    )

    print("Alle Plots wurden im Ordner 'iso_plots' gespeichert!")