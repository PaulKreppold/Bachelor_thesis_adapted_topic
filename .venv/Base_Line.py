import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumModel(nn.Module):
    def __init__(
        self,
        num_qubits=6,
        num_layers=4,
        init_mode="identity"
    ):
        super().__init__()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.init_mode = init_mode

        dev = qml.device("default.qubit", wires=num_qubits)
        # Weight Shape bleibt (L, Q, 3) für qml.Rot
        weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            for l in range(self.num_layers):
                # 1. Data Re-uploading (Dense Angle Encoding)
                # Bleibt erhalten für Nicht-Linearität
                for i in range(self.num_qubits):
                    qml.RY(inputs[:, i], wires=i)
                    qml.RZ(inputs[:, i + self.num_qubits], wires=i)

                # 2. Variational Block (Rotations)
                for i in range(self.num_qubits):
                    qml.Rot(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=i
                    )

                # 3. STRONGLY ENTANGLING STRUCTURE
                # Der Shift bestimmt, welches Qubit mit welchem verschränkt wird.
                # Er ändert sich pro Layer: shift = 1, 2, 3...
                shift = (l % (self.num_qubits - 1)) + 1
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + shift) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Initialisierung bleibt wie von dir gewünscht
        if self.init_mode == "identity":
            nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)
        elif self.init_mode == "uniform":
            nn.init.uniform_(self.qlayer.weights, a=-np.pi, b=np.pi)
        else:
            raise ValueError(f"Unbekannter init_mode: {self.init_mode}")

    def forward(self, x):
        # x shape: (batch, 12)
        q_out = self.qlayer(x)
        return q_out.unsqueeze(1)