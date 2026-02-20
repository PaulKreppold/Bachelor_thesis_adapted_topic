import pennylane as qml
import torch.nn as nn
import torch

class QuantumModel(nn.Module):
    def __init__(self, num_qubits, num_layers, ansatz="hardware_efficient"):
        super().__init__()
        self.n_qubits = num_qubits
        self.n_layers = num_layers
        self.ansatz = ansatz
        self.dev = qml.device("default.qubit", wires=num_qubits)

        if ansatz == "strongly_entangling":
            self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        else:
            self.weight_shapes = {"weights": (num_layers, num_qubits)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # 1. State Preparation (Amplitude Embedding)
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True, pad_with=0.0)

            # 2. Variational Layers
            if self.ansatz == "strongly_entangling":
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            else:
                for l in range(num_layers):
                    for q in range(num_qubits):
                        qml.RY(weights[l, q], wires=q)
                    for q in range(num_qubits):
                        qml.CNOT(wires=[q, (q + 1) % num_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

    def forward(self, x):
        if x.ndim > 2: x = x.view(x.size(0), -1)
        q_out = self.qlayer(x)
        q_out_mean = torch.mean(q_out, dim=1)
        return ((q_out_mean + 1) / 2).view(-1, 1)