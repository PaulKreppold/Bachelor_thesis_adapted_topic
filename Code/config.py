import os
import platform
import multiprocessing
import torch

# =============================================================================
# 1. THREADING & CPU OPTIMIERUNG (MUSS GANZ OBEN STEHEN)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# =============================================================================
# 2. SYSTEM-ERKENNUNG & PFADE
# =============================================================================
IS_MAC = platform.system() == "Darwin"
IS_AWS_CLOUD = True

DEVICE = torch.device("cpu")

HOME = os.path.expanduser("~")
PROJECT_PATH = "/home/ec2-user/Code"
DATASETS_DIR = os.path.join(PROJECT_PATH, "datasets")

RSNA_ROOT = os.path.join(DATASETS_DIR, "rsna_28.npz")
CHEXPERT_ROOT = os.path.join(DATASETS_DIR, "chexpert_28.npz")
BASE_RESULTS_DIR = os.path.join(PROJECT_PATH, "final_results")

S3_BUCKET_NAME = "paul-bachelor-thesis-data"
S3_PREFIX = "final_results/REPRESENTATIVE_MATRIX_RUN/"

MAX_WORKERS = 15
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# =============================================================================
# 3. HYPERPARAMETER (Inkl. TWO_QUBIT_NOISE_FACTOR)
# =============================================================================
CALIBRATION_SEED = 42
RANDOM_SEEDS = [42, 117, 52, 31, 10]
PROBS = [0.1, 0.05, 0.03, 0.01]

NUM_QUBITS = 4
NUM_FEATURES = NUM_QUBITS * 2
TWO_QUBIT_NOISE_FACTOR = 5.0
DEFAULT_LAYERS = 4

BATCH_SIZE = 16
LEARNING_RATE = 0.001
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 3
BASELINE_EPOCHS = GLOBAL_ROUNDS * LOCAL_EPOCHS

ALL_SUBCLIENTS = [f"client_{c}_sub_{s}" for c in range(1, 5) for s in range(1, 5)]

# =============================================================================
# 4. MISSING DATA MATRIX (Alle verbleibenden 109 Tasks)
# =============================================================================
MISSING_DATA = {
    # Physics ISO (Uniform)
    "ISO_PHASE_FLIP_p0.1": [42, 117, 52],
    "ISO_PHASE_FLIP_p0.05": [42, 31, 117, 10, 52],
    "ISO_PHASE_FLIP_p0.03": [117, 52],
    "ISO_PHASE_FLIP_p0.01": [42, 31, 117, 10, 52],

    "ISO_PHASE_DAMPING_p0.1": [42, 31, 117, 10],
    "ISO_PHASE_DAMPING_p0.05": [42, 31, 117, 10, 52],
    "ISO_PHASE_DAMPING_p0.03": [31],
    "ISO_PHASE_DAMPING_p0.01": [42, 31, 117, 10],

    "ISO_DEPOLARIZING_p0.1": [42, 10],
    "ISO_DEPOLARIZING_p0.05": [31, 117, 10, 52],
    "ISO_DEPOLARIZING_p0.03": [42, 31, 117, 52],
    "ISO_DEPOLARIZING_p0.01": [117, 52, 10],

    "ISO_BIT_FLIP_p0.1": [42, 31, 117, 52],
    "ISO_BIT_FLIP_p0.05": [31, 117, 52, 10],
    "ISO_BIT_FLIP_p0.03": [42, 31, 117, 10],
    "ISO_BIT_FLIP_p0.01": [42, 31, 52, 10],

    "ISO_AMPLITUDE_DAMPING_p0.1": [42, 10, 52],
    "ISO_AMPLITUDE_DAMPING_p0.05": [31, 52],
    "ISO_AMPLITUDE_DAMPING_p0.03": [42, 31, 10, 52],
    "ISO_AMPLITUDE_DAMPING_p0.01": [117],

    # Bias Core (Gate Dependent)
    "BIAS_GLOBAL_NOISE_p0.1": [42, 31, 117, 10, 52],
    "BIAS_GLOBAL_NOISE_p0.05": [42, 31, 117, 10, 52],
    "BIAS_GLOBAL_NOISE_p0.03": [42, 31, 117, 10, 52],
    "BIAS_GLOBAL_NOISE_p0.01": [42, 31, 117, 10, 52],

    "BIAS_TRAP_p0.1": [42, 31, 52],
    "BIAS_TRAP_p0.05": [42, 10, 52],
    "BIAS_TRAP_p0.03": [117, 10],
    "BIAS_TRAP_p0.01": [117, 52],

    "BIAS_REALISM_p0.1": [42, 117, 10],
    "BIAS_REALISM_p0.05": [42, 31, 52],
    "BIAS_REALISM_p0.03": [31, 117],
    "BIAS_REALISM_p0.01": [42, 10],

    "BIAS_OUTLIER_DEPOLARIZING_p0.1": [31, 117],
    "BIAS_OUTLIER_DEPOLARIZING_p0.05": [42, 10, 52],
    "BIAS_OUTLIER_AMPLITUDE_DAMPING_p0.1": [42, 10],

    "MODE_CHECK_UNIFORM": [31, 117, 10, 52],
    "MODE_CHECK_GATE_DEPENDENT": [42, 31, 117, 10, 52]
}

ALL_SCENARIOS = []


def add_s(group, label, cfg):
    if label in MISSING_DATA:
        cfg.update({
            "group": group,
            "label": label,
            "specific_seeds": MISSING_DATA[label],
            "layers": cfg.get('layers', DEFAULT_LAYERS),
            "noise_mode": cfg.get('noise_mode', 'uniform')
        })
        ALL_SCENARIOS.append(cfg)


# --- 5. SZENARIEN BEFÜLLUNG ---

# Physics ISO
for nt in ["depolarizing", "amplitude_damping", "phase_damping", "bit_flip", "phase_flip"]:
    for p in PROBS:
        add_s("physics_iso", f"ISO_{nt.upper()}_p{p}", {"type": "iso", "noise_type": nt, "p": p})

# Bias Core & Outliers
for p in PROBS:
    add_s("bias_core", f"BIAS_GLOBAL_NOISE_p{p}", {
        "type": "hetero", "noise_mode": "gate_dependent",
        "mapping": {sid: ("depolarizing", p) for sid in ALL_SUBCLIENTS}
    })
    add_s("bias_core", f"BIAS_TRAP_p{p}", {
        "type": "hetero", "noise_mode": "gate_dependent",
        "mapping": {"client_1": ("depolarizing", p), "client_4": ("depolarizing", p)}
    })
    add_s("bias_core", f"BIAS_REALISM_p{p}", {
        "type": "hetero", "noise_mode": "gate_dependent",
        "mapping": {
            "client_1": ("depolarizing", p), "client_2": ("amplitude_damping", p),
            "client_3": ("phase_damping", p), "client_4": (None, 0.0)
        }
    })

for p_out in [0.05, 0.1]:
    add_s("bias_core", f"BIAS_OUTLIER_DEPOLARIZING_p{p_out}", {
        "type": "hetero", "noise_mode": "gate_dependent",
        "mapping": {"client_1": ("depolarizing", p_out)}
    })
add_s("bias_core", "BIAS_OUTLIER_AMPLITUDE_DAMPING_p0.1", {
    "type": "hetero", "noise_mode": "gate_dependent",
    "mapping": {"client_1": ("amplitude_damping", 0.1)}
})

# Ablations
add_s("ablation", "MODE_CHECK_UNIFORM",
      {"type": "iso", "noise_type": "depolarizing", "p": 0.03, "noise_mode": "uniform"})
add_s("ablation", "MODE_CHECK_GATE_DEPENDENT",
      {"type": "iso", "noise_type": "depolarizing", "p": 0.03, "noise_mode": "gate_dependent"})

ALL_SCENARIOS.sort(key=lambda x: x.get('p', 0.0), reverse=True)

if __name__ == "__main__":
    print(f"✅ Config GELADEN. {len(ALL_SCENARIOS)} Szenarien identifiziert.")
    total_tasks = sum(len(s['specific_seeds']) for s in ALL_SCENARIOS)
    print(f"📊 Insgesamt zu berechnende Tasks (Seeds): {total_tasks}")