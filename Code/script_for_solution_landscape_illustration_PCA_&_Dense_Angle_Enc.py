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
# Basispfad zu deinen PCA-Ergebnissen
BASE_PATH = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic/Thesis_Results/REF_NOISELESS_L4"
OUTPUT_DIR = "pca_landscape_plots"
CSV_PATH = os.path.join(OUTPUT_DIR, "pca_barren_plateau_analysis.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter aus deiner Config
SEEDS = [42, 117, 52, 31, 10]
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FEATURES = 8  # 4 Qubits * 2 (RY + RZ)

RESOLUTION = 50
COLORMAP = 'viridis'


# ==========================================
# 2. HILFSFUNKTION: PCA + DENSE ANGLE LANDSCAPE
# ==========================================
def get_pca_landscape(weights, seed):
    dev = qml.device("default.qubit", wires=NUM_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, w):
        # --- DENSE ANGLE ENCODING (identisch zu deiner Klasse) ---
        for q in range(NUM_QUBITS):
            qml.RY(inputs[q], wires=q)
            qml.RZ(inputs[q + NUM_QUBITS], wires=q)

        # --- VARIATIONAL LAYERS ---
        qml.StronglyEntanglingLayers(w, wires=range(NUM_QUBITS))

        return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

    # Richtungsvektoren fixieren
    torch.manual_seed(seed)
    d1 = torch.randn_like(weights);
    d1 /= torch.norm(d1)
    d2 = torch.randn_like(weights);
    d2 /= torch.norm(d2)

    # Dummy Input (8 Features für PCA Dense Angle)
    # Wir nehmen Werte im Bereich [0, pi], um typische Winkel darzustellen
    sample_input = torch.tensor([np.pi / 4] * NUM_FEATURES)

    coords = np.linspace(-1.0, 1.0, RESOLUTION)
    Z = np.zeros((RESOLUTION, RESOLUTION))

    # Fortschrittsbalken
    pbar = tqdm(total=RESOLUTION * RESOLUTION, desc=f"   Grid S{seed}", leave=False)
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            with torch.no_grad():
                w_p = weights + coords[i] * d1 + coords[j] * d2
                out = circuit(sample_input, w_p)
                Z[i, j] = (torch.mean(torch.stack(out)) + 1) / 2
            pbar.update(1)
    pbar.close()
    return coords, Z


# ==========================================
# 3. STATISTISCHE ANALYSE (Alle Seeds)
# ==========================================
def run_full_analysis():
    print(f"🚀 Starte Analyse für PCA + Dense Angle ({NUM_QUBITS} Qubits, {NUM_LAYERS} Layer)")
    results = []

    for s in SEEDS:
        path = os.path.join(BASE_PATH, "models", f"S{s}", "central_baseline.pt")
        if not os.path.exists(path):
            print(f"⚠️ Datei fehlt: {path}")
            continue

        state_dict = torch.load(path, map_location="cpu")
        # Bei deiner PCA-Klasse liegen die Gewichte unter 'qlayer.weights'
        w_opt = state_dict['qlayer.weights']

        coords, Z = get_pca_landscape(w_opt, s)

        results.append({
            "Seed": s,
            "Variance": np.var(Z),
            "Max": np.max(Z),
            "Min": np.min(Z)
        })

        # Visualisierung nur für Seed 42 speichern
        if s == 42:
            save_visuals(coords, Z, s)

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Statistik unter {CSV_PATH} gespeichert.")


# ==========================================
# 4. VISUALISIERUNG (3D & HEATMAP)
# ==========================================
def save_visuals(coords, Z, seed):
    X, Y = np.meshgrid(coords, coords)

    # --- HEATMAP ---
    plt.figure(figsize=(8, 6))
    im = plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap=COLORMAP)
    plt.colorbar(im, label="Modell Output")
    plt.title(f"Heatmap: PCA + Dense Angle (Seed {seed})")
    plt.xlabel("Richtung 1")
    plt.ylabel("Richtung 2")
    plt.savefig(f"{OUTPUT_DIR}/PCA_Heatmap_S{seed}.pdf", bbox_inches='tight')
    plt.close()

    # --- 3D PLOT ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=COLORMAP, antialiased=True, linewidth=0)
    ax.set_zlim(0, 1)
    ax.set_title(f"3D: PCA + Dense Angle (Seed {seed})")
    plt.savefig(f"{OUTPUT_DIR}/PCA_3D_Plot_S{seed}.pdf", bbox_inches='tight')
    plt.close()


# ==========================================
# 5. FINALER VARIANZ-PLOT
# ==========================================
def plot_final_variance():
    df = pd.read_csv(CSV_PATH)
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="whitegrid")
    # Da wir hier nur ein Szenario haben (PCA 4Q), nutzen wir einen Barplot
    sns.barplot(data=df, y='Variance', color="teal", errorbar='sd')
    plt.title("Varianz der PCA-Loss-Landschaft")
    plt.ylabel(r"Varianz $\sigma^2$")
    plt.savefig(f"{OUTPUT_DIR}/PCA_Variance_Stability.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_full_analysis()
    plot_final_variance()
    print(f"✅ Alle PCA-Plots wurden in '{OUTPUT_DIR}' erstellt.")