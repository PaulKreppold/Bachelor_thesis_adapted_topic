import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=8, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        # StronglyEntanglingLayers: (Layer, Qubits, 3 Rotationen)
        self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # State Preparation
            qml.AmplitudeEmbedding(
                inputs,
                wires=range(self.num_qubits),
                normalize=True, pad_with=0.0
            )

            # Strongly Entangling Layers
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(self.num_qubits)
            )

            # Messung
            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # Initialisierung
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)

    def forward(self, x):
        # 16x16 → 256
        x = x.view(-1, 784)

        q_out = self.qlayer(x)

        # [-1, 1] → [0, 1]
        probs = (q_out + 1) / 2

        return probs.unsqueeze(1)

