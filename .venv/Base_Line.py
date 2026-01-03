import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import config


class QuantumModel(nn.Module):
    def __init__(
            self,
            num_qubits=config.NUM_QUBITS,
            num_layers=config.NUM_LAYERS,
            init_mode="identity",
            encoding_mode="RY_RZ"
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.encoding_mode = encoding_mode
        self.init_mode = init_mode

        num_feat = self.num_qubits * 2
        dev = qml.device("default.qubit", wires=self.num_qubits)

        # Shapes für TorchLayer
        self.weight_shapes = {
            "weights": (self.num_layers, self.num_qubits),
            "s_alpha": (num_feat,),
            "s_beta": (num_feat,)
        }

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights, s_alpha, s_beta):
            # Transformation: x' = alpha * x + beta
            # WICHTIG: Wir nutzen torch.clamp oder halten die Werte klein,
            # um die Periodizität zu kontrollieren.
            transformed_inputs = s_alpha * inputs + s_beta

            for l in range(self.num_layers):
                # 1. Data Re-Uploading
                for i in range(self.num_qubits):
                    f1 = transformed_inputs[:, i]
                    f2 = transformed_inputs[:, i + self.num_qubits]

                    if self.encoding_mode == "RY_RZ":
                        qml.RY(f1, wires=i)
                        qml.RZ(f2, wires=i)
                    elif self.encoding_mode == "RX_RY":
                        qml.RX(f1, wires=i)
                        qml.RY(f2, wires=i)

                # 2. Variational Layer (HEA)
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)

                # 3. Entanglement
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # --- STABILE INITIALISIERUNG (Entscheidend!) ---
        with torch.no_grad():
            # Alpha muss bei 1.0 starten (Identität)
            self.qlayer.s_alpha.fill_(1.0)
            # Beta muss bei 0.0 starten
            self.qlayer.s_beta.fill_(0.0)

            if self.init_mode == "identity":
                # Gewichte nahe 0, damit der Layer anfangs "durchsichtig" ist
                nn.init.normal_(self.qlayer.weights, mean=0.0, std=0.01)

    def forward(self, x):
        q_out = self.qlayer(x)
        return q_out.unsqueeze(1)