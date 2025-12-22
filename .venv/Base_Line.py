import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=10, num_layers=6, use_scaling=False, ansatz_type="StronglyEntangling"):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_scaling = use_scaling
        self.ansatz_type = ansatz_type

        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Dynamische Gewichtsformen basierend auf dem Ansatz
        if self.ansatz_type == "StronglyEntangling":
            self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        elif self.ansatz_type == "EfficientSU2":
            # RY und RZ Rotationen pro Qubit -> 2 Parameter pro Qubit/Layer
            self.weight_shapes = {"weights": (num_layers, num_qubits, 2)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), pad_with=0.0, normalize=True)

            if self.ansatz_type == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

            elif self.ansatz_type == "EfficientSU2":
                for layer_weights in weights:
                    # Rotation Layer (ähnlich VQC6 aus dem Paper)
                    for i in range(self.num_qubits):
                        qml.RY(layer_weights[i, 0], wires=i)
                        qml.RZ(layer_weights[i, 1], wires=i)
                    # Entanglement Layer (Circular CNOTs)
                    for i in range(self.num_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        if self.use_scaling:
            self.scale = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('bias', torch.zeros(1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        q_out = self.qlayer(x)
        logits = (q_out * self.scale) + self.bias
        return logits.view(-1, 1)