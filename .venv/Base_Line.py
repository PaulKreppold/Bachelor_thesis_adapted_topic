import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=10, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        self.weight_shapes = {"weights": (num_layers, num_qubits)}
        self.qnode = self._create_qnode()

        self.quantum_layer = qml.qnn.TorchLayer(
            self.qnode,
            self.weight_shapes,
            init_method=lambda w: torch.nn.init.uniform_(w, 0, 2 * np.pi)
        )

    def _create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. Angle Embedding
            qml.AngleEmbedding(features=inputs, wires=range(self.num_qubits), rotation='Y')

            # 2. Basic Entangler Layers
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RY(weights[l, q], wires=q)
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            # Messung ALLER Qubits -> Liste von [-1, 1] Werten
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit

    def forward(self, x):
        # Input x: [batch_size, 10]
        q_out = self.quantum_layer(x) # [batch_size, 10]

        # WICHTIG FÜR MSE: Wir nutzen den Mittelwert (Mean).
        # Ergebnisbereich: [-1.0, 1.0].
        # Das passt perfekt zu Labels, die wir auf -1 und 1 mappen.
        output = torch.mean(q_out, dim=1, keepdim=True)

        return output