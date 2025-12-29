import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pfad-Konfiguration (Lokal)
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/VQC_ablation_study_AmplitudeEmbedding_on_binary_MNIST'

# Hyperparameter
SEEDS = [42]#, 1337, 2024, 7, 101]
BATCH_SIZE = 32
NUM_QUBITS = 10
EPOCHS_PER_EXP = 100
EARLY_STOPPING_PATIENCE = 12

# Studien-Parameter
STUDY_LAYERS = [2, 4, 6, 8]
STUDY_LRS = [0.01, 0.005, 0.001]#, 0.0005, 0.0001]
STUDY_MEASURE_MODES = [False, True] # False=Single, True=All (Mean)
STUDY_LOSS_MODES = ["MSE", "BCE"]