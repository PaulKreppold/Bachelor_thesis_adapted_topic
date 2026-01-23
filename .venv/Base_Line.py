import torch
import torch.nn as nn
import pennylane as qml

class QuantumModel(nn.Module):
    def __init__(
        self,
        num_qubits,
        num_layers,
        encoding="amplitude"  # "amplitude" | "angle_pca"
    ):
        super().__init__()

        assert encoding in ["amplitude", "angle_pca"]
        self.encoding = encoding
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.dev = qml.device("default.qubit", wires=num_qubits)

        self.weight_shapes = {
            "weights": (num_layers, num_qubits, 3)
        }

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):

            # ---------------------------
            # 1. ENCODING
            # ---------------------------
            if self.encoding == "amplitude":
                qml.AmplitudeEmbedding(
                    inputs,
                    wires=range(self.num_qubits),
                    normalize=True
                )
            elif self.encoding == "angle_pca":
                # inputs.shape = (2 * num_qubits,)
                for q in range(self.num_qubits):
                    qml.RY(inputs[q], wires=q)
                    qml.RZ(inputs[q + self.num_qubits], wires=q)

            # ---------------------------
            # 2. VARIATIONAL ANSATZ
            # ---------------------------
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(self.num_qubits)
            )

            # ---------------------------
            # 3. READOUT
            # ---------------------------
            return qml.expval(qml.PauliZ(0))

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        # Xavier Initialization mit gain=0.1
        nn.init.xavier_normal_(self.qlayer.weights['weights'].data, gain=0.1)

    def forward(self, x):
        # Robust gegen verschiedene Input-Shapes
        if x.ndim == 4:  # (B, C, H, W)
            x = x.view(x.shape[0], -1)
        elif x.ndim == 2:  # (B, features)
            pass
        elif x.ndim == 1:  # einzelnes Sample
            x = x.unsqueeze(0)

        q_out = self.qlayer(x)
        return ((q_out + 1) / 2).view(-1, 1)



