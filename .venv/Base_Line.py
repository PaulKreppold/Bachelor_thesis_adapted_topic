import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=10, num_layers=6):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.dev = qml.device("default.qubit", wires=num_qubits)

        self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(
                features=inputs,
                wires=range(self.num_qubits),
                pad_with=0.0,
                normalize=True
            )

            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Flattening der Bilder (1x28x28 auf 784 bzw. 1x32x32 auf 1024)
        x = x.view(x.size(0), -1)

        # Werte zwischen -1 und 1
        q_out = self.qlayer(x)

        # lineare Skalierung erlaubt dem Modell die Entscheidungsgrenze (0) zu verschieben & Konfidenz/Wahrscheinlichkeit zu erhöhen
        logits = (q_out * self.scale) + self.bias

        return logits.view(-1, 1)