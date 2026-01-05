import torch
import os

# --- System & Pfade ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/new_ablation_study_for_PCA_&_AngleEncoding'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Statistik & Training ---
SEEDS = [42, 1337, 2024]
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.001

# --- ABLATION SCENARIOS ---
# Erläuterung der Parameter:
# 'weighted': Bestraft Fehler bei Klasse 0 (Gesund) stärker (hilft gegen 0.4-Loss-Plateau).
# 'trainable_enc': Modell lernt Feature-Importance (w*x + b).
# 'measurement': 'mean' (lokale Observablen) vs 'first' (globale Observable).
ABLATION_SCENARIOS = [
    # 1. Baseline & Qubit-Variation
    {"name": "Q4_Base", "qubits": 4, "features": 48, "layers": 1, "encoding": "dense", "type": "strongly_entangling",
     "reuploading": True, "weighted": False, "trainable_enc": False, "measurement": "mean"},
    {"name": "Q6_Base", "qubits": 6, "features": 48, "layers": 1, "encoding": "dense", "type": "strongly_entangling",
     "reuploading": True, "weighted": False, "trainable_enc": False, "measurement": "mean"},
    {"name": "Q8_Base", "qubits": 8, "features": 48, "layers": 1, "encoding": "dense", "type": "strongly_entangling",
     "reuploading": True, "weighted": False, "trainable_enc": False, "measurement": "mean"},

    # 2. Re-Uploading Check (Fair Comparison)
    {"name": "Q6_NoReupload_12f", "qubits": 6, "features": 12, "layers": 1, "encoding": "dense",
     "type": "strongly_entangling", "reuploading": False, "weighted": False, "trainable_enc": False,
     "measurement": "mean"},
    {"name": "Q6_StandardEnc_48f", "qubits": 6, "features": 48, "layers": 1, "encoding": "standard",
     "type": "strongly_entangling", "reuploading": True, "weighted": False, "trainable_enc": False,
     "measurement": "mean"},

    # 3. Mess-Strategie (Global vs. Lokal)
    {"name": "Q6_Measure_FirstOnly", "qubits": 6, "features": 48, "layers": 1, "encoding": "dense",
     "type": "strongly_entangling", "reuploading": True, "weighted": False, "trainable_enc": False,
     "measurement": "first"},

    # 4. Die "Loss-Knacker" (Ziel: < 0.4 Loss)
    {"name": "Q6_WeightedLoss", "qubits": 6, "features": 48, "layers": 1, "encoding": "dense",
     "type": "strongly_entangling", "reuploading": True, "weighted": True, "trainable_enc": False,
     "measurement": "mean"},
    {"name": "Q6_TrainableEncoding", "qubits": 6, "features": 48, "layers": 1, "encoding": "dense",
     "type": "strongly_entangling", "reuploading": True, "weighted": False, "trainable_enc": True,
     "measurement": "mean"},

    # 5. Maximal-Modell
    {"name": "Q6_L2_Weighted_Trainable", "qubits": 6, "features": 48, "layers": 2, "encoding": "dense",
     "type": "strongly_entangling", "reuploading": True, "weighted": True, "trainable_enc": True,
     "measurement": "mean"},
]