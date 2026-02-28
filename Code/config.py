import torch
import os
import platform

# --- SYSTEM & SCHEDULING ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAC = platform.system() == "Darwin"
MAX_WORKERS = 4 if IS_MAC else 80

if IS_MAC:
    PROJECT_ROOT = "/Users/paulkreppold/python_projects/SWM/Bachelor_thesis_adapted_topic"
    RSNA_ROOT = os.path.join(PROJECT_ROOT, "datasets/rsna_28.npz")
    CHEXPERT_ROOT = os.path.join(PROJECT_ROOT, "datasets/chexpert_28.npz")
    BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "Thesis_Results")
else:
    HOME = os.path.expanduser("~")
    PROJECT_PATH = os.path.join(HOME, "Bachelor_thesis_adapted_topic")
    RSNA_ROOT = os.path.join(PROJECT_PATH, "datasets/rsna_28.npz")
    CHEXPERT_ROOT = os.path.join(PROJECT_PATH, "datasets/chexpert_28.npz")
    BASE_RESULTS_DIR = os.path.join(HOME, "final_results")

os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# --- HYPERPARAMETER ---
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

ALL_SCENARIOS = []
def add_s(group, label, cfg):
    cfg.update({"group": group, "label": label})
    ALL_SCENARIOS.append(cfg)

# =============================================================================
# 1. REFERENCE (4) - Deine Baseline ("Ideale Welt")
# =============================================================================
for l in [8, 6, 4, 2]:

    add_s("reference", f"REF_NOISELESS_L{l}",
          {"type": "base", "layers": l, "noise_type": None, "p": 0.0})

# =============================================================================
# 2. PHYSICS ISO (20 Szenarien) - Jetzt mit Bit- und Phase-Flip
# =============================================================================
noise_types = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "bit_flip",    # NEU
    "phase_flip"   # NEU
]

for nt in noise_types:
    for p in PROBS: # [0.01, 0.03, 0.05, 0.1]
        add_s("physics_iso", f"ISO_{nt.upper()}_p{p}",
              {"type": "iso", "noise_type": nt, "p": p, "layers": 4, "noise_mode": "uniform"})

# =============================================================================
# 3. BIAS CORE (10) - Dein Hauptbeitrag (LDS & Noise Mix)
# =============================================================================
# Hier nutzen wir deine 90/40/60 Verteilung aus data_prep voll aus!

for p in PROBS:
    # BIAS GLOBAL: Alle 16 Kliniken (90%, 40%, 60%) rauschen gleichmäßig
    add_s("bias_core", f"BIAS_GLOBAL_NOISE_p{p}",
          {"type": "hetero", "layers": 4, "noise_mode": "gate_dependent",
           "mapping": {sid: ("depolarizing", p) for sid in ALL_SUBCLIENTS}})

    # BIAS TRAP: Nur Klinikgruppen mit hohem LDS (75-90%) haben Rauschen.
    # Betrifft 8 Subclients: Jeweils sub_1 (90%) und sub_2 bis sub_4 (75%).
    add_s("bias_core", f"BIAS_TRAP_p{p}",
          {"type": "hetero", "layers": 4, "noise_mode": "gate_dependent",
           "mapping": {"client_1": ("depolarizing", p), "client_4": ("depolarizing", p)}})

    # CLINICAL REALISM: Jede Klinik-Gruppe hat ein anderes Rausch-Problem
    add_s("bias_core", f"BIAS_REALISM_p{p}",
          {"type": "hetero", "layers": 4, "noise_mode": "gate_dependent",
           "mapping": {
               "client_1": ("depolarizing", p),      # Klinik 1: Weißes Rauschen
               "client_2": ("amplitude_damping", p), # Klinik 2: Energieverlust
               "client_3": ("phase_damping", p),     # Klinik 3: Dekohärenz
               "client_4": (None, 0.0)               # Klinik 4: Perfekte Hardware
           }})

# OUTLIER STRESS (4) - Was passiert, wenn eine Klinik (z.B. Klinik 1) extrem rauscht?
for p_out in [0.05, 0.1]:
    for nt_out in ["depolarizing", "amplitude_damping"]:
        add_s("bias_core", f"BIAS_OUTLIER_{nt_out.upper()}_p{p_out}",
              {"type": "hetero", "layers": 4, "noise_mode": "gate_dependent",
               "mapping": {"client_1": (nt_out, p_out)}})

# =============================================================================
# 4. ABLATIONS (4) - Methodik-Check
# =============================================================================
for mode in ["uniform", "gate_dependent"]:
    add_s("ablation", f"MODE_CHECK_{mode.upper()}",
          {"type": "iso", "noise_type": "depolarizing", "p": 0.03, "layers": 4, "noise_mode": mode})

add_s("ablation", "MIXED_REALISM_STRESS",
      {"type": "hetero", "layers": 4, "noise_mode": "gate_dependent",
       "mapping": {"client_2": ("depolarizing", 0.05), "client_3": ("phase_damping", 0.05)}})

# --- FINAL VALIDATION ---
if __name__ == "__main__":
    print(f"✅ Loaded {len(ALL_SCENARIOS)} scenarios.")
