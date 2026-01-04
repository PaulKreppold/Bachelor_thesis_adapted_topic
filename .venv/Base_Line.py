import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits=6, num_layers=6, encoding="angle", use_scaling=True):
        """
        Variational Quantum Classifier (VQC)

        Args:
            num_qubits (int): Anzahl der Qubits (standardmäßig 6).
            num_layers (int): Tiefe des Variational Circuits.
            encoding (str): "angle" für Dense Angle Encoding oder "iqp" für IQP-style Mapping.
            use_scaling (bool): Wenn True, werden lernbare Scale- und Bias-Parameter hinzugefügt.
        """
        super().__init__()
        self.n_qubits = num_qubits
        self.n_layers = num_layers
        self.encoding = encoding
        self.use_scaling = use_scaling

        # Device definieren
        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Gewichte-Shape für StronglyEntanglingLayers: (L, Q, 3)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # ==========================================
            # 1. ENCODING LAYER (Daten-Input)
            # ==========================================
            if self.encoding == "iqp":
                # IQP-Style: Superposition -> RZ -> IsingZZ Interaktionen
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)

                # Linearer Teil (Features 0 bis 5)
                for i in range(self.n_qubits):
                    qml.RZ(inputs[:, i], wires=i)

                # Interaktions-Teil (Features 6 bis 11)
                for i in range(self.n_qubits):
                    qml.IsingZZ(inputs[:, i + self.n_qubits], wires=[i, (i + 1) % self.n_qubits])

            else:  # Standard: Dense Angle Encoding
                # Features 0-5 auf RY, Features 6-11 auf RZ
                for i in range(self.n_qubits):
                    qml.RY(inputs[:, i], wires=i)
                    qml.RZ(inputs[:, i + self.n_qubits], wires=i)

            # ==========================================
            # 2. VARIATIONAL LAYER (Lernbare Gewichte)
            # ==========================================
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Messung des ersten Qubits (PauliZ liefert Werte zwischen -1 und 1)
            return qml.expval(qml.PauliZ(0))

        # PennyLane QNN Layer
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # ==========================================
        # 3. OUTPUT SCALING (Optional)
        # ==========================================
        if self.use_scaling:
            # Initiale Skalierung auf 5.0 hilft dem BCE-Loss, aus dem 0.5-Plateau zu kommen
            self.scale = nn.Parameter(torch.tensor([5.0]))
            self.bias = nn.Parameter(torch.tensor([0.0]))

        # Xavier/Glorot Initialisierung der Quanten-Gewichte
        with torch.no_grad():
            for param in self.qlayer.parameters():
                nn.init.xavier_normal_(param, gain=0.1)

    def forward(self, x):
        """
        Input x: (Batch_Size, 12)
        Output:  (Batch_Size, 1) Logits für BCEWithLogitsLoss
        """
        # q_out Form: (Batch_Size,)
        q_out = self.qlayer(x)

        # Umwandeln in (Batch_Size, 1) für PyTorch Loss-Kompatibilität
        q_out = q_out.reshape(-1, 1)

        if self.use_scaling:
            # Transformation: y = w * x + b
            return q_out * self.scale + self.bias

        return q_out

