import pennylane as qml
import torch
import torch.nn as nn


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=10, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        # NEU: 2 Parameter pro Qubit pro Layer (für RY und RZ)
        self.weight_shapes = {"weights": (num_layers, num_qubits, 2)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # 1. DENSE ANGLE ENCODING (LaRose & Coyle)
            for i in range(self.num_qubits):
                qml.RY(inputs[:, i], wires=i)
                qml.RZ(inputs[:, i + self.num_qubits], wires=i)

            # 2. VARIATIONAL LAYERS (Erweitert um RZ)
            for l in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i, 0], wires=i)  # Erster Parameter: RY
                    qml.RZ(weights[l, i, 1], wires=i)  # Zweiter Parameter: RZ

                # Entanglement (Ring-Struktur)
                if self.num_qubits > 1:
                    for i in range(self.num_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)

    def forward(self, x):
        q_out = self.qlayer(x)
        return q_out.unsqueeze(1)