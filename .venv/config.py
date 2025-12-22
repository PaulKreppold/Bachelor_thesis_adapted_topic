import torch

# --- Device Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Clients ---
clients = ["client_1", "client_2", "client_3"]

client_qubits = {client: 10 for client in clients}

# --- Hyperparameters ---
SEEDS = [42, 1337, 2024, 112, 19]
NUM_EPOCHS = 25         # VQC braucht oft länger, oder weniger je nach Konvergenz
LEARNING_RATE = 0.01    # 0.01 ist oft besser für Adam bei VQCs als 0.001
BATCH_SIZE = 32         # Wie besprochen
NUM_LAYERS = 6          # Tiefe des VQC





