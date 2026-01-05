import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/Umfangreiche_Ablation_AngleEncoding'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 1337, 2024]
BATCH_SIZE = 32
NUM_EPOCHS = 40

# --- UMFANGREICHE ABLATION SCENARIOS ---
ABLATION_SCENARIOS = [
    # PHASE 1: Architektur-Isolation (Basis: Q6, L2, Dense, HE, Ring, First, LR 0.001)
    {"name": "P1_Baseline_HE_Ring_First", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P1_Var_Encoding_Standard", "qubits": 6, "layers": 2, "encoding": "standard", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P1_Var_Ansatz_Strongly", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "strongly", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P1_Var_Entanglement_AllToAll", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "all_to_all", "measurement": "first", "lr": 0.001},
    {"name": "P1_Var_Measure_Mean", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "mean", "lr": 0.001},
    {"name": "P1_Var_Measure_Softmax", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "softmax", "lr": 0.001},

    # PHASE 2: Skalierungs-Matrix (Beispiel-Auswahl aus 4x4 Qubit/Layer Matrix)
    {"name": "P2_Q4_L2", "qubits": 4, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P2_Q4_L8", "qubits": 4, "layers": 8, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P2_Q8_L4", "qubits": 8, "layers": 4, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},
    {"name": "P2_Q10_L8", "qubits": 10, "layers": 8, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.001},

    # PHASE 3: Hyperparameter
    {"name": "P3_LR_0.01", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.01},
    {"name": "P3_LR_0.005", "qubits": 6, "layers": 2, "encoding": "dense", "ansatz": "hardware_efficient", "entanglement": "ring", "measurement": "first", "lr": 0.005},
]
