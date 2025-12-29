import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=10, num_layers=2, measure_all=False, use_bce=False):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.measure_all = measure_all
        self.use_bce = use_bce
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weight_shapes = {"weights": (num_layers, num_qubits, 2)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), pad_with=0.0, normalize=True)
            for layer in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Circular CNOT Ring (Verschränkung)
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            if self.measure_all:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            else:
                return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # --- INITIALISIERUNG: Kleine Gewichte gegen Barren Plateaus ---
        torch.nn.init.uniform_(self.qlayer.weights, a=-0.01, b=0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        q_out = self.qlayer(x)

        if self.measure_all:
            q_out = torch.mean(q_out, dim=1, keepdim=True)
        else:
            q_out = q_out.view(-1, 1)

        if self.use_bce:
            return torch.sigmoid(q_out)  # Wichtig für BCELoss
        return q_out