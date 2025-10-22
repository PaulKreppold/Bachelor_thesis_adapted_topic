import torch
import torch.nn as nn
import pennylane as qml

class QuantumModel(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumModel, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        self.weight_shapes = {"params": (self.num_layers, self.num_qubits, 3)}

        self.qnode = self.create_qnode()
        self.quantum_layer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes, init_method=lambda w: torch.nn.init.normal_(w, mean=0.0, std=0.01))


    def create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            qml.templates.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True, pad_with=0.0)

            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)

                if l % 2 == 0:
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                else:
                    for q in range(self.num_qubits - 1):
                        qml.CZ(wires=[q, q + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit


    def forward(self, inputs):
        output = self.quantum_layer(inputs)
        output = torch.mean(output, dim=1)
        return output.view(-1,1) * 5.0
