import torch
import pennylane as qml
import numpy as np
import torch.nn as nn


class QuantumModel(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumModel, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Parameter für VQC
        self.weight_shapes = {"weights": (self.num_layers, self.num_qubits, 3)}

        self.qnode = self.create_qnode()

        # Quanten Layer init über vollen Raum [0, 2π])
        self.quantum_layer = qml.qnn.TorchLayer(
            self.qnode,
            self.weight_shapes,
            init_method=lambda w: torch.nn.init.uniform_(w, 0, 2 * np.pi)
        )

    def create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                normalize=True
            )

            qml.StronglyEntanglingLayers(
                weights=weights,
                wires=range(self.num_qubits)
            )

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        # Quantum Layer: Outputs in [-1, +1] (batch, num_qubits)
        out = self.quantum_layer(x)

        # Mean über alle Qubits: (batch, 1)
        out = out.mean(dim=1, keepdim=True)

        # Sigmoid anwenden: [-1, +1] → [0, 1] Wahrscheinlichkeiten
        out = torch.sigmoid(out)

        return out