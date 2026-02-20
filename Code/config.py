import os
import platform
import torch

# =============================================================================
# 1. THREADING & OPTIMIERUNG (Für Mac Performance)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Verhindert, dass Bibliotheken wie NumPy eigenständig alle Kerne belegen
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# =============================================================================
# 2. SYSTEM-ERKENNUNG & DYNAMISCHE PFADE
# =============================================================================
IS_MAC = platform.system() == "Darwin"
IS_AWS_CLOUD = False

DEVICE = torch.device("cpu")

# Pfad-Logik für deine Struktur:
# 1. CODE_DIR ist dort, wo diese config.py liegt (.../Code/)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. PROJECT_ROOT ist eine Ebene höher (.../Bachelor_thesis_adapted_topic/)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
# 3. DATASETS_DIR ist im Projekt-Stammverzeichnis (.../datasets/)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Absolute Pfade zu den Dateien
RSNA_ROOT = os.path.join(DATASETS_DIR, "rsna_28.npz")
CHEXPERT_ROOT = os.path.join(DATASETS_DIR, "chexpert_28.npz")

# Ergebnisse werden im Code-Ordner gespeichert
BASE_RESULTS_DIR = os.path.join(CODE_DIR, "ablation_results_Amplitude_Enc")
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# DEINE PERFORMANCE KERNE: Festlegung auf 4 parallele Prozesse
MAX_WORKERS = 4

# =============================================================================
# 3. BASIS-HYPERPARAMETER
# =============================================================================
CALIBRATION_SEED = 42
RANDOM_SEEDS = [42, 117, 52, 31, 10]

BATCH_SIZE = 16
LEARNING_RATE = 0.001
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 3
BASELINE_EPOCHS = GLOBAL_ROUNDS * LOCAL_EPOCHS

ALL_SUBCLIENTS = [f"client_{c}_sub_{s}" for c in range(1, 5) for s in range(1, 5)]

# =============================================================================
# 4. ABLATIONS-MATRIX (Fokus auf Q4-Layer-Vergleich + Q10-Upper-Bound)
# =============================================================================
ABLATION_QUBITS = [10]  # 4 für Layer-Check, 10 für Barren-Plateau-Beweis
ABLATION_LAYERS = [4, 6]
ABLATION_ANSATZ = ["hardware_efficient", "strongly_entangling"]

ALL_SCENARIOS = []

for q in ABLATION_QUBITS:
    # Bei 10 Qubits testen wir nur 2 und 8 Layer, um Zeit zu sparen (reicht als Beweis)
    current_layers = ABLATION_LAYERS #if q == 4 else [2, 8]

    for l in current_layers:
        for a in ABLATION_ANSATZ:
            ALL_SCENARIOS.append({
                "group": "amplitude_ablation",
                "label": f"AE_{a}_Q{q}_L{l}",
                "num_qubits": q,
                "num_layers": l,
                "ansatz": a,
                "type": "noiseless",
                "noise_type": None,
                "p": 0.0,
                "specific_seeds": RANDOM_SEEDS
            })

# =============================================================================
# 5. DIAGNOSE-AUSGABE
# =============================================================================
if __name__ == "__main__":
    print(f"✅ Ablation-Config geladen.")
    print(f"📂 Datasets gesucht in: {DATASETS_DIR}")
    print(f"📄 RSNA vorhanden: {os.path.exists(RSNA_ROOT)}")
    print(f"🚀 Worker (Kerne): {MAX_WORKERS}")
    total_tasks = len(ALL_SCENARIOS) * len(RANDOM_SEEDS)
    print(f"📊 Gesamtzahl der Simulationen: {total_tasks}")