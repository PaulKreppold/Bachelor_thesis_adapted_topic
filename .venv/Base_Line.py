import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=8, num_layers=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        # nur RY pro Qubit & Layer
        self.weight_shapes = {"weights": (num_layers, num_qubits)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # State Preparation
            qml.AmplitudeEmbedding(
                inputs,
                wires=range(self.num_qubits),
                normalize=True,
                pad_with=0.0
            )

            # Ansatz: RY + CNOT-Ring
            for l in range(self.num_layers):

                # RY Rotationen
                for q in range(self.num_qubits):
                    qml.RY(weights[l, q], wires=q)

                # CNOT Ring
                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

            # Messung aller Qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.1)

    def forward(self, x):
        x = x.view(-1, 256)

        # [batch_size, num_qubits]
        q_out = self.qlayer(x)

        # Mean über alle Qubits
        q_mean = q_out.mean(dim=1)

        # [-1, 1] → [0, 1]
        probs = (q_mean + 1) / 2

        return probs.unsqueeze(1)



