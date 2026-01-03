import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/VQC_PCA_DenseEncoding_Pneumonia'
SEEDS = [42, 1337, 2024]#, 7, 101]
BATCH_SIZE = 32
NUM_QUBITS = 6
NUM_FEATURES = NUM_QUBITS * 2 # 10 Qubits * 2 (RY & RZ)
NUM_EPOCHS = 100
NUM_LAYERS = 2
LR = 0.01

# Definition der Grid-Profile: (Name, Qubits, Layers)
GRID_PROFILES = [
    {"name": "Q6_L2", "qubits": 6, "layers": 2}
]
ABLATION_ENCODINGS = ["RY_RZ", "RX_RY", "RX_RZ"]