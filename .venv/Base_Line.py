import pennylane as qml
import torch
import torch.nn as nn


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=6, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        # NEU: Nur 1 Parameter (RY) pro Qubit pro Layer
        self.weight_shapes = {"weights": (num_layers, num_qubits)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)

            for l in range(self.num_layers):
                # 1. Rotations-Schicht
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)

                # 2. Verschränkungs-Schicht (Ring-Struktur)
                if self.num_qubits > 1:
                    for i in range(self.num_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # WICHTIG: Initialisierung näher bei 0
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)

    def forward(self, x):
        # Flattening: (Batch, 1, 8, 8) -> (Batch, 64)
        x = x.view(-1, 64)

        # Quantum Layer Output
        q_out = self.qlayer(x)

        # Shape Anpassung für MSE [Batch, 1]
        return q_out.unsqueeze(1)
