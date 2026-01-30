import torch
import os

# =============================================================================
# SYSTEM & PFADE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anzahl der parallelen Prozesse (Seeds), die gleichzeitig laufen sollen
MAX_WORKERS = 2  # Ermöglicht die parallele Ausführung von 2 Seeds

RSNA_ROOT = "/Users/paulkreppold/bachelor_thesis/Local_datasets/RSNA_Pneumonia_Detection_Challenge"
CHEXPERT_ROOT = "/Users/paulkreppold/bachelor_thesis/Local_datasets/CheXpert_Frontal_Processed"
BASE_RESULTS_DIR = "/Users/paulkreppold/bachelor_thesis/Phase_1_isolation_study"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# =============================================================================
# FEDERATED STRUKTUR
# =============================================================================
BASE_CLIENTS = ["client_1", "client_2", "client_3", "client_4"]
NUM_SUBCLIENTS_PER_BASE = 4
ALL_SUBCLIENTS = [f"{base}_sub_{i}" for base in BASE_CLIENTS for i in range(1, NUM_SUBCLIENTS_PER_BASE + 1)]

# =============================================================================
# TRAINING-PARAMETER
# =============================================================================
SEEDS = [42, 117]#, 52, 31]  # Aktuell auf 2 Seeds für Tests begrenzt
BATCH_SIZE = 16
LR = 0.001
QFL_GLOBAL_ROUNDS = 2
QFL_LOCAL_EPOCHS = 2
EPOCHS_BASELINE = QFL_GLOBAL_ROUNDS * QFL_LOCAL_EPOCHS

# =============================================================================
# QUANTUM MODELL ARCHITEKTUR
# =============================================================================
NUM_QUBITS = 2
NUM_LAYERS = 2
NUM_FEATURES = NUM_QUBITS * 2 # 2 Features (RY, RZ) pro Qubit (Dense Angle Encoding)

# =============================================================================
# NOISE PARAMETER (NISQ-ÄRA UNTERSUCHUNG)
# =============================================================================
NOISE_PROBS = [0.01, 0.05]#, 0.1]  # Rauschwahrscheinlichkeiten
NOISE_TYPES = ["bitflip", "phaseflip", "depolarizing"] # Rauscharten [cite: 16, 25]

# Theoretische Definition der Experiment-Szenarien
PHASE1_SCENARIOS = [("None", 0.0)] + [(nt, np) for nt in NOISE_TYPES for np in NOISE_PROBS if np > 0]