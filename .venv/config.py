import torch
import os

# System & Pfade
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RSNA_ROOT = "/Users/paulkreppold/bachelor_thesis/Local_datasets/RSNA_Pneumonia_Detection_Challenge"
CHEXPERT_ROOT = "/Users/paulkreppold/bachelor_thesis/Local_datasets/CheXpert_Frontal_Processed"
BASE_RESULTS_DIR = "/Users/paulkreppold/bachelor_thesis/Phase_1_isolation_study"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# Federated Struktur
BASE_CLIENTS = ["client_1", "client_2", "client_3", "client_4"]
NUM_SUBCLIENTS_PER_BASE = 4
ALL_SUBCLIENTS = [f"{base}_sub_{i}" for base in BASE_CLIENTS for i in range(1, NUM_SUBCLIENTS_PER_BASE + 1)]

# Training-Parameter
SEEDS = [42, 117, 52, 19]
BATCH_SIZE = 16
LR = 0.001
QFL_GLOBAL_ROUNDS = 5
QFL_LOCAL_EPOCHS = 3
EPOCHS_BASELINE = QFL_GLOBAL_ROUNDS * QFL_LOCAL_EPOCHS

# Quantum Modell Architektur
NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_FEATURES = NUM_QUBITS * 2 # 2 Features (RY, RZ) pro Qubit

# Noise Parameter
NOISE_PROBS = [0.01, 0.05, 0.1]
NOISE_TYPES = ["bitflip", "phaseflip", "depolarizing"]
PHASE1_SCENARIOS = [("None", 0.0)] + [(nt, np) for nt in NOISE_TYPES for np in NOISE_PROBS if np > 0]