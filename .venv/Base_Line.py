import pennylane as qml
import torch.nn as nn
import torch
import config


class QuantumModelNoiseless(nn.Module):
    def __init__(self, n_layers, encoding_type="dense", measurement_type="first"):
        super().__init__()
        self.n_qubits = config.NUM_QUBITS
        self.n_layers = n_layers
        self.encoding_type = encoding_type  # "normal", "dense", "triple"
        self.measurement_type = measurement_type  # "first", "all_mean"

        # Für die reine Baseline ohne Rauschen nutzen wir default.mixed (Konsistenz)
        self.dev = qml.device("default.mixed", wires=self.n_qubits)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        def circuit(inputs, weights):
            # -------- 1. ENCODING OPTIONEN --------
            # Normal: 4 Features (62.13% Varianz)
            if self.encoding_type == "normal":
                for q in range(self.n_qubits):
                    qml.RY(inputs[q], wires=q)

            # Dense: 8 Features (74.61% Varianz)
            elif self.encoding_type == "dense":
                for q in range(self.n_qubits):
                    qml.RY(inputs[q], wires=q)
                    qml.RZ(inputs[q + self.n_qubits], wires=q)

            # Triple: 12 Features (80.35% Varianz)
            elif self.encoding_type == "triple":
                for q in range(self.n_qubits):
                    qml.RX(inputs[q], wires=q)
                    qml.RY(inputs[q + self.n_qubits], wires=q)
                    qml.RZ(inputs[q + 2 * self.n_qubits], wires=q)

            # -------- 2. VARIATIONAL ANSATZ --------
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # -------- 3. MESS-STRATEGIE --------
            if self.measurement_type == "first":
                # Nur ein Qubit (Standard-Ansatz)
                return qml.expval(qml.PauliZ(0))
            else:
                # Alle Qubits (Mittelwertbildung folgt im Forward)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            # Xavier Initialisierung für stabilen Gradientenfluss
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        if x.ndim == 1: x = x.unsqueeze(0)

        # Ausführung des Quanten-Layers
        try:
            q_out = self.qlayer(x)
        except Exception:
            q_out = torch.stack([self.qlayer(sample) for sample in x])

        # Mittelwertbildung falls measurement_type="all_mean"
        if self.measurement_type == "all_mean":
            # q_out Form: (Batch, n_qubits) -> Mean über dim 1
            q_out = torch.mean(q_out, dim=1, keepdim=True)

        # Mapping von Erwartungswert [-1, 1] auf Wahrscheinlichkeit [0, 1]
        return ((q_out.flatten() + 1) / 2).view(-1, 1)