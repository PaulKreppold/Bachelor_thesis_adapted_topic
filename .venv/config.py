import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/VQC_PCA_Dense_AngleEncoding_second_test'

# Statistik & Training
SEEDS = [42, 1337, 2024]
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 0.001

# Quanten-Architektur
NUM_QUBITS = 6
NUM_FEATURES = NUM_QUBITS * 2
NUM_LAYERS = 6

# Experiment-Konfigurationen (Die 2x2 Matrix)
EXPERIMENT_CONFIGS = [
    {"encoding": "angle", "use_scaling": False},
    {"encoding": "angle", "use_scaling": True},
    {"encoding": "iqp",   "use_scaling": False},
    {"encoding": "iqp",   "use_scaling": True},
]
