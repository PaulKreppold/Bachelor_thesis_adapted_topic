import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_final_ablation(base_path, scenarios, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    epochs = np.arange(1, 26)  # Erwartet 25 Epochen pro Seed

    print(f"Suche Daten in: {base_path}")

    for scenario_dir, label in zip(scenarios, labels):
        path = os.path.join(base_path, scenario_dir)
        acc_list = []
        loss_list = []

        if not os.path.exists(path):
            print(f"⚠️ Pfad nicht gefunden: {path}")
            continue

        # Lade alle JSON-Dateien im Szenario-Ordner
        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        for jf in json_files:
            with open(os.path.join(path, jf), 'r') as f:
                data = json.load(f)
                acc_list.append(data['metrics']['train_acc'])
                loss_list.append(data['metrics']['train_loss'])

        if acc_list:
            print(f"✅ Verarbeite {len(acc_list)} Seeds für: {label}")
            # Berechnung von Mittelwert und Standardabweichung
            acc_array = np.array(acc_list)
            loss_array = np.array(loss_list)

            m_acc, s_acc = np.mean(acc_array, axis=0), np.std(acc_array, axis=0)
            m_loss, s_loss = np.mean(loss_array, axis=0), np.std(loss_array, axis=0)

            # Plot Accuracy (Linie + Shading)
            line, = ax1.plot(epochs, m_acc, label=label, linewidth=2.5)
            ax1.fill_between(epochs, m_acc - s_acc, m_acc + s_acc, color=line.get_color(), alpha=0.15)

            # Plot Loss (Linie + Shading)
            line_l, = ax2.plot(epochs, m_loss, label=label, linewidth=2.5)
            ax2.fill_between(epochs, m_loss - s_loss, m_loss + s_loss, color=line_l.get_color(), alpha=0.15)

    # Styling Accuracy
    ax1.set_title('Trainingsgenauigkeit (Mittelwert ± Std)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochen', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(False)  # Kein Raster

    # Styling Loss
    ax2.set_title('Trainingsverlust (Mittelwert ± Std)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochen', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(False)  # Kein Raster

    plt.tight_layout()
    plt.savefig('final_ablation_comparison.png', dpi=300)
    plt.show()


# Absolute Pfade und Szenarien
BASE_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Code/ablation_results_final"

scenarios = [
    '1_Suenkel_Sanity_MNIST',
    '2_Suenkel_Baseline_AE',
    '3_AE_MaxExpressive_12L',
    '4_AE_InfoLoss_4Q',
    '5_PCA_Baseline_4Q',
    '6_PCA_Final_Target',      # Komma hier war wichtig!
    '7_PCA_HighCapacity_10Q_12L'
]

labels = [
    "MNIST: AE (Basistest)",
    "AE: 10Q, HEA (L=15)",
    "AE: 10Q, SVS (L=12)", # Der "alte" High-End Versuch
    "AE: 4Q, HEA (L=4)",
    "PCA+DWK: 4Q, SVS (L=4)", # Dein finales Rausch-Modell
    "PCA+DWK: 10Q, SVS (L=4)",
    "PCA+DWK: 10Q, SVS (L=12)" # Der "faire Vergleich" zu Label 3
]

plot_final_ablation(BASE_PATH, scenarios, labels)