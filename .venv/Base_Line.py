import torch
import torch.nn as nn
import pennylane as qml
import config


class UniversalVQC(nn.Module):
    def __init__(self, scenario):
        super().__init__()
        self.n_qubits = scenario['qubits']
        self.n_layers = scenario['layers']
        self.encoding = scenario['encoding']
        self.ansatz = scenario['ansatz']
        self.entanglement = scenario['entanglement']
        self.measurement = scenario['measurement']

        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Weight Shapes bestimmen
        if self.ansatz == 'strongly':
            weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        else:
            weight_shapes = {"weights": (self.n_layers, self.n_qubits)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # 1. ENCODING (Standard oder Dense)
            for q in range(self.n_qubits):
                qml.RY(inputs[:, q], wires=q)
                if self.encoding == 'dense':
                    qml.RZ(inputs[:, q + self.n_qubits], wires=q)

            # 2. ANSATZ (HE oder Strongly)
            if self.ansatz == 'strongly':
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                for l in range(self.n_layers):
                    for q in range(self.n_qubits):
                        qml.RY(weights[l, q], wires=q)
                    # Entanglement Topologie
                    if self.entanglement == 'all_to_all':
                        for i in range(self.n_qubits):
                            for j in range(i + 1, self.n_qubits):
                                qml.CNOT(wires=[i, j])
                    else:  # Ring
                        for q in range(self.n_qubits):
                            qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

            # 3. MEASUREMENT
            if self.measurement == 'softmax':
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
            elif self.measurement == 'mean':
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            else:  # First
                return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        q_out = self.qlayer(x)
        if self.measurement == 'softmax':
            return torch.softmax((q_out + 1) / 2, dim=1)
        elif self.measurement == 'mean':
            res = torch.mean(q_out, dim=1)
            return ((res + 1) / 2).reshape(-1, 1)
        else:
            return ((q_out + 1) / 2).reshape(-1, 1)