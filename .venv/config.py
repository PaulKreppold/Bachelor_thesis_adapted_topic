import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pfad-Konfiguration (Lokal)
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/VQC_ablation_study_AmplitudeEmbedding_on_binary_MNIST'

# Hyperparameter
SEEDS = [42, 1337, 2024, 7, 101]
BATCH_SIZE = 32
NUM_QUBITS = 10
NUM_EPOCHS = 50
NUM_LAYERS = 8
LR = 0.001