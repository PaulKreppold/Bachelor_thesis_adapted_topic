import pennylane as qml
import torch
import torch.nn as nn


class QuantumModel(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumModel, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # KORREKTUR 1: Weight Shapes anpassen
        # BasicEntanglerLayers braucht nur 1 Parameter pro Qubit pro Layer
        # Shape: (num_layers, num_qubits)
        self.weight_shapes = {"params": (self.num_layers, self.num_qubits)}

        self.qnode = self.create_qnode()
        self.quantum_layer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes,
                                                init_method=lambda w: torch.nn.init.normal_(w, mean=0.0, std=0.05))

    def create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            qml.templates.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True, pad_with=0.0)

            # KORREKTUR 2: qml.RY statt 'Y' übergeben
            qml.templates.BasicEntanglerLayers(params, wires=range(self.num_qubits), rotation=qml.RY)

            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, inputs):
        # inputs shape: [batch, 28, 28] -> muss flach sein
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)

        exp_val = self.quantum_layer(inputs)

        # Umrechnung von Erwartungswert [-1, 1] in Wahrscheinlichkeit [0, 1]
        # exp_val = +1 (|0>) -> prob = 0
        # exp_val = -1 (|1>) -> prob = 1
        prob = (1 - exp_val) / 2

        # Sicherstellen, dass die Dimension [batch, 1] ist für BCELoss
        return prob.unsqueeze(1).float()