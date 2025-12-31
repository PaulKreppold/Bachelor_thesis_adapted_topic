import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=8, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        # 2 Parameter (RY und RZ) pro Qubit pro Layer
        self.weight_shapes = {"weights": (num_layers, num_qubits, 2)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # 1. State Preparation
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)

            for l in range(self.num_layers):
                # 1. Rotations-Schicht
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

                # 2. Verschränkung: "Linear Entanglement" statt Ring
                # Ein Ring-Muster kann bei vielen Qubits schneller zu Barren Plateaus führen.
                # Ein lineares Muster (i auf i+1) ist oft "lokaler" und stabiler.
                if self.num_qubits > 1:
                    for i in range(self.num_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            # WICHTIGSTE ÄNDERUNG: Lokale Observablen statt Globaler Observable
            # Statt nur PauliZ(0) zu messen (global), messen wir alle Qubits einzeln (lokal)
            # und mitteln das Ergebnis. Das erhöht die Gradientensignale massiv.
            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # Initialisierung: Extreme Nähe zu 0 (Identity-Initialization)
        # Wir starten fast bei einer Identitäts-Transformation, um Gradienten zu erhalten.
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.01)

    def forward(self, x):
        x = x.view(-1, 64)

        q_out = self.qlayer(x)

        return q_out.unsqueeze(1)
