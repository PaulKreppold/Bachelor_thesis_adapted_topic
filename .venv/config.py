import torch
import os


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# 2. Pfade
BASE_PATH = "/Users/paulkreppold/bachelor_thesis/results/prestudy_angle_pca"
PLOT_DIR = os.path.join(BASE_PATH, "plots")
STATS_DIR = os.path.join(BASE_PATH, "stats")
MODEL_DIR = os.path.join(BASE_PATH, "models")

for d in [PLOT_DIR, STATS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. Hyperparameter
NUM_QUBITS = 4
MAX_FEATURES = 12
BATCH_SIZE = 16
LR = 0.01
EPOCHS_BASELINE = 10
SEEDS = [42, 123]

# 4. Ablation Setup (18 Konfigurationen)
LAYERS_LIST = [2, 4, 6]
ENCODINGS = ["normal", "dense", "triple"]
MEASUREMENTS = ["first", "all_mean"]

ABLATION_CONFIGS = []
for l in LAYERS_LIST:
    for enc in ENCODINGS:
        for meas in MEASUREMENTS:
            ABLATION_CONFIGS.append({"L": l, "enc": enc, "meas": meas})