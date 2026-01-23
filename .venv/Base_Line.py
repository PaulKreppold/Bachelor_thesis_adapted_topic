import pennylane as qml
import torch.nn as nn
import torch
import config


class QuantumModel(nn.Module):
    def __init__(self, n_layers, noise_type=None, p=0.0):
        super().__init__()
        self.n_qubits = config.NUM_QUBITS
        self.n_layers = n_layers
        self.noise_type = noise_type
        self.p = p

        # Wir nutzen default.mixed für Rausch-Simulationen
        self.dev = qml.device("default.mixed", wires=self.n_qubits)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        # Definieren der QNode ohne den Dekorator auf der Funktion
        def circuit(inputs, weights):
            # Encoding: RY und RZ
            for q in range(self.n_qubits):
                qml.RY(inputs[q], wires=q)
                qml.RZ(inputs[q + self.n_qubits], wires=q)

            # Variabler Ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Noise Channels
            if self.p > 0:
                for q in range(self.n_qubits):
                    if self.noise_type == "bitflip":
                        qml.BitFlip(self.p, wires=q)
                    elif self.noise_type == "phaseflip":
                        qml.PhaseFlip(self.p, wires=q)
                    elif self.noise_type == "depolarizing":
                        qml.DepolarizingChannel(self.p, wires=q)

            return qml.expval(qml.PauliZ(0))

        # Hier die QNode erstellen
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        # WICHTIG: Wir nutzen TorchLayer, aber wir sagen ihm,
        # dass er die QNode direkt nutzen soll.
        # Das automatische Broadcasting wird hier oft durch TorchLayer erzwungen.
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        # Falls x eine Batch-Dimension fehlt (z.B. bei Einzelwerten)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Wir versuchen den direkten Aufruf.
        # Falls default.mixed bei Rauschen immer noch meckert,
        # nutzen wir einen einfachen Loop (langsam aber sicher für 16er Batches)
        try:
            q_out = self.qlayer(x)
        except Exception:
            # Fallback: Falls Broadcasting bei Mixed States fehlschlägt,
            # berechnen wir die Samples einzeln.
            # Das ist der sicherste Weg für Mixed State Noise Simulationen.
            q_out = torch.stack([self.qlayer(sample) for sample in x])

        return ((q_out.flatten() + 1) / 2).view(-1, 1)