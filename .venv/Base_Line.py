import torch
import torch.nn as nn
import pennylane as qml

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

        # Nur noch 1 Parameter (RY) pro Qubit pro Layer
        # Shape: (Anzahl Layer, Anzahl Qubits)
        self.weight_shapes = {"weights": (num_layers, num_qubits)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # 1. State Preparation (256 Features für 8 Qubits bei 16x16 Bildern)
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)

            for l in range(self.num_layers):
                # 1. Rotations-Schicht: Nur RY
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)

                # 2. Verschränkung (Entanglement)
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            # Messung aller Qubits (Lokale Observablen) zur Stabilisierung des Gradienten
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # Initialisierung mit erhöhter Standardabweichung für bessere Exploration
        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)

    def forward(self, x):
        # 16x16 Pixel = 256 Amplituden
        x = x.view(-1, 256)

        # Erwartungswerte berechnen
        q_out = self.qlayer(x)

        # Mittelwert über alle Qubits bilden (falls Batch-Dimension vorhanden)
        if q_out.dim() > 1:
            q_out = torch.mean(q_out, dim=1)

        # Mapping von [-1, 1] auf [0, 1] für die Wahrscheinlichkeitsinterpretation
        probs = (q_out + 1) / 2

        return probs.unsqueeze(1)
