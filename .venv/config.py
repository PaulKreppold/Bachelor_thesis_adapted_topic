import torch
import os

# System & Pfade
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/Super_Model_layer_and_init_ablation_study'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training-Parameter
SEEDS = [42, 1337, 2024]
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.001

# Fixierte Super-Modell Architektur
NUM_QUBITS = 10
layer_configs = [4, 6, 8, 10, 12]
init_methods = ["xavier", "kaiming", "small_normal", "uniform_2pi"]
# Standard Encoding nutzt 1 Feature pro Qubit (RY)
NUM_FEATURES = NUM_QUBITS * 2