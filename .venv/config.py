import torch
import os

# System & Pfade
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = '/Users/paulkreppold/bachelor_thesis/VQC_PCA_DataReUploading_48feats'

# Verzeichnis erstellen, falls nicht vorhanden
os.makedirs(RESULTS_DIR, exist_ok=True)

# Statistik & Training
SEEDS = [42, 1337, 2024]
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 0.001  # Eventuell auf 0.005 - 0.01 erhöhen, da kein Scaling-Parameter vorhanden ist

# Quanten-Architektur (Data Re-Uploading Setup)
NUM_QUBITS = 6
NUM_FEATURES = 48  # Erhöht auf 48 Komponenten für bessere Bild-Repräsentation

# Bestimmt, wie viele Features pro Qubit-Paar (RY/RZ) geladen werden
FEATURES_PER_BLOCK = NUM_QUBITS * 2

# Anzahl der Upload-Zyklen (hier: 48 / 12 = 4 Blöcke)
NUM_BLOCKS = NUM_FEATURES // FEATURES_PER_BLOCK

# Variational Layers pro Block (entspricht der Tiefe nach jedem Upload)
LAYERS_PER_BLOCK = 4

# Hinweis: Die Gesamtzahl der trainierbaren Schichten ist NUM_BLOCKS * LAYERS_PER_BLOCK