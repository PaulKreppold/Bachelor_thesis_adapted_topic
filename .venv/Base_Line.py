import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import config

class QuantumModel(nn.Module):
    def __init__(self, n_layers, init_method):
        super().__init__()
        self.n_qubits = config.NUM_QUBITS
        self.n_layers = n_layers  # Dynamisch aus dem Loop
        self.init_method = init_method

        dev = qml.device("default.qubit", wires=self.n_qubits)
        # StronglyEntanglingLayers erwartet (L, Q, 3) Weights
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # 1. DENSE ENCODING
            for q in range(self.n_qubits):
                qml.RY(inputs[:, q], wires=q)
                qml.RZ(inputs[:, q + self.n_qubits], wires=q)

            # 2. ANSATZ
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # 3. MEASUREMENT
            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # --- Initialisierungs-Logik ---
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            if self.init_method == "xavier":
                nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)
            elif self.init_method == "kaiming":
                # Kaiming für ReLU-Verhalten optimiert
                nn.init.kaiming_normal_(self.qlayer.weights, mode='fan_out', nonlinearity='relu')
            elif self.init_method == "small_normal":
                # Sehr kleine Gewichte zur Vermeidung früher Barren Plateaus
                nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.01)
            elif self.init_method == "uniform_2pi":
                # Klassische Quanten-Initialisierung
                nn.init.uniform_(self.qlayer.weights, a=0, b=2*np.pi)
            else:
                raise ValueError(f"Unbekannte Init-Methode: {self.init_method}")

    def forward(self, x):
        q_out = self.qlayer(x)
        # Transformation von [-1, 1] auf [0, 1] für BCELoss
        return ((q_out + 1) / 2).reshape(-1, 1)