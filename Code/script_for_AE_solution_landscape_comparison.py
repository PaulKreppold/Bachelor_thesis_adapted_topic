import torch
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 1. KONFIGURATION & PFADE
# ==========================================
BASE_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Code/ablation_results_Amplitude_Enc"
# Pfad zur CSV, die du bereits erstellt hast
CSV_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Code/landscape_exports/barren_plateau_analysis.csv"
OUTPUT_DIR = "scientific_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter für die Visualisierung
VIS_SEED = 42
QUBITS = [4, 10]
ANSATZES = ["hardware_efficient", "strongly_entangling"]
NUM_LAYERS = 4
RESOLUTION = 50  # Erhöht für glatte Grafiken
COLORMAP = 'viridis'


# ==========================================
# 2. HILFSFUNKTION FÜR DIE BERECHNUNG
# ==========================================
def get_landscape(n_q, anz, seed):
    folder = f"AE_{anz}_Q{n_q}_L{NUM_LAYERS}"
    path = os.path.join(BASE_PATH, folder, "models", f"S{seed}", "central_baseline.pt")
    if not os.path.exists(path):
        print(f"⚠️ Datei fehlt: {path}")
        return None, None

    state_dict = torch.load(path, map_location="cpu")
    weights = state_dict['qlayer.weights']

    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        qml.AmplitudeEmbedding(inputs, wires=range(n_q), normalize=True, pad_with=0.0)
        if anz == "strongly_entangling":
            qml.StronglyEntanglingLayers(w, wires=range(n_q))
        else:
            for l in range(NUM_LAYERS):
                for q in range(n_q): qml.RY(w[l, q], wires=q)
                for q in range(n_q): qml.CNOT(wires=[q, (q + 1) % n_q])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    torch.manual_seed(seed)
    d1 = torch.randn_like(weights);
    d1 /= torch.norm(d1)
    d2 = torch.randn_like(weights);
    d2 /= torch.norm(d2)
    inp = torch.ones(2 ** n_q) / np.sqrt(2 ** n_q)
    coords = np.linspace(-1.0, 1.0, RESOLUTION)
    Z = np.zeros((RESOLUTION, RESOLUTION))

    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            with torch.no_grad():
                w_p = weights + coords[i] * d1 + coords[j] * d2
                out = circuit(inp, w_p)
                Z[i, j] = (torch.mean(torch.stack(out)) + 1) / 2
    return coords, Z


# ==========================================
# 3. PLOT 1: VARIANZ-ANALYSE (DER BEWEIS)
# ==========================================
def plot_variance_analysis():
    print("📊 1/3: Generiere Varianz-Analyse (Statistischer Beweis)...")
    df = pd.read_csv(CSV_PATH)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Barplot mit Error-Bars für alle Seeds
    ax = sns.barplot(data=df, x='Qubits', y='Variance', hue='Ansatz', errorbar='sd', palette=COLORMAP)
    ax.set_yscale('log')  # Log-Skala ist essenziell für Barren Plateaus

    plt.title('Wissenschaftlicher Nachweis: Barren Plateau Effekt', fontsize=14, pad=15)
    # Hier das 'r' vor dem String gegen den Syntax-Error:
    plt.ylabel(r'Varianz der Kostenfunktion $\sigma^2$ (log-Skala)')
    plt.xlabel('Anzahl der Qubits')

    plt.savefig(f"{OUTPUT_DIR}/1_variance_statistical_proof.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# 4. PLOT 2: HEATMAP VERGLEICH (TOPOLOGIE)
# ==========================================
def plot_heatmap_grid():
    print("🗺️ 2/3: Generiere Heatmap-Vergleich (2x2 Grid)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for r, n_q in enumerate(QUBITS):
        for c, anz in enumerate(ANSATZES):
            _, Z = get_landscape(n_q, anz, VIS_SEED)
            if Z is not None:
                # vmin/vmax sorgt für eine einheitliche Farbskala
                im = axes[r, c].imshow(Z, extent=[-1, 1, -1, 1], cmap=COLORMAP,
                                       origin='lower', vmin=0.45, vmax=0.65)
                axes[r, c].set_title(f"{anz.replace('_', ' ').title()}\n{n_q} Qubits", fontsize=12)
                fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)

            axes[r, c].set_xlabel("Richtung 1")
            axes[r, c].set_ylabel("Richtung 2")

    plt.suptitle(f"Topologischer Vergleich der Verlustlandschaften (Seed {VIS_SEED})", fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/2_heatmap_comparison.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# 5. PLOT 3: REPRÄSENTATIVE 3D ANSICHT
# ==========================================
def plot_3d_representative():
    print("⛰️ 3/3: Generiere 3D-Visualisierung (Q4 vs Q10)...")
    anz = "strongly_entangling"  # Wir fokussieren uns auf einen Ansatz
    for n_q in QUBITS:
        coords, Z = get_landscape(n_q, anz, VIS_SEED)
        if Z is None: continue

        X, Y = np.meshgrid(coords, coords)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=COLORMAP, antialiased=True,
                               rcount=RESOLUTION, ccount=RESOLUTION, linewidth=0)

        ax.set_zlim(0, 1)  # Konsistente Höhe für den Vergleich
        ax.set_title(f"3D Ansicht: {anz.replace('_', ' ').title()} ({n_q} Qubits)", pad=20)
        ax.set_xlabel("Richtung 1")
        ax.set_ylabel("Richtung 2")
        ax.set_zlabel("Modell Output")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.savefig(f"{OUTPUT_DIR}/3_3d_comparison_Q{n_q}.pdf", bbox_inches='tight')
        plt.close()


# ==========================================
# START
# ==========================================
if __name__ == "__main__":
    plot_variance_analysis()
    plot_heatmap_grid()
    plot_3d_representative()
    print(f"\n✅ Alle Grafiken wurden im Ordner '{OUTPUT_DIR}' gespeichert.")