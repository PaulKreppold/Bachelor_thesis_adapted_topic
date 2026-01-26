import torch
import torch.nn as nn
import pennylane as qml

class QuantumModel(nn.Module):
    def __init__(
        self,
        num_qubits,
        num_layers,
        encoding="amplitude",        # "amplitude" | "angle_pca"
        ansatz="strongly"             # "strongly" | "hardware_efficient"
    ):
        super().__init__()

        assert encoding in ["amplitude", "angle_pca"]
        assert ansatz in ["strongly", "hardware_efficient"]

        self.encoding = encoding
        self.ansatz = ansatz
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Einheitliche Weight-Form (wichtig für Vergleichbarkeit)
        self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        # -----------------------------
        # QNode definieren
        # -----------------------------
        def circuit(inputs, weights):

            # -------- Encoding --------
            if self.encoding == "amplitude":
                qml.AmplitudeEmbedding(
                    inputs,
                    wires=range(self.num_qubits),
                    normalize=True
                )

            elif self.encoding == "angle_pca":
                for q in range(self.num_qubits):
                    qml.RY(inputs[q], wires=q)
                    qml.RZ(inputs[q + self.num_qubits], wires=q)

            # -------- Ansatz --------
            if self.ansatz == "strongly":
                qml.StronglyEntanglingLayers(
                    weights, wires=range(self.num_qubits)
                )

            elif self.ansatz == "hardware_efficient":
                for l in range(self.num_layers):
                    for q in range(self.num_qubits):
                        qml.RY(weights[l, q, 0], wires=q)
                        qml.RZ(weights[l, q, 1], wires=q)

                    # lineare Entanglement-Topologie
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

            return qml.expval(qml.PauliZ(0))

        # QNode
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        # TorchLayer
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

        # Initialisierung
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        # x: (B, C, H, W) oder (B, features)
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)

        outputs = []
        for i in range(x.shape[0]):
            outputs.append(self.qlayer(x[i]))

        q_out = torch.stack(outputs)

        # Map [-1, 1] → [0, 1]
        return ((q_out + 1) / 2).view(-1, 1)






