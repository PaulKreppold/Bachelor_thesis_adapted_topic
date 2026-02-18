import pennylane as qml
import torch.nn as nn
import torch
import config
from noise_utils import build_noise_model


class QuantumModel(nn.Module):
    def __init__(self, n_layers, noise_type=None, p=0.0, noise_mode="uniform",
                 two_qubit_factor=config.TWO_QUBIT_NOISE_FACTOR):
        super().__init__()
        self.n_qubits = config.NUM_QUBITS
        self.n_layers = n_layers
        self.noise_type = noise_type
        self.p = p
        self.noise_mode = noise_mode
        self.two_qubit_factor = two_qubit_factor

        # --- 1. DYNAMISCHER SIMULATOR-SWITCH ---
        # Vereinfacht: Nur None und p=0 checken
        if noise_type is None or p == 0.0:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            self.dev = qml.device("default.mixed", wires=self.n_qubits)

        # --- 2. SCHALTKREIS-DEFINITION ---
        def circuit(inputs, weights):
            # Encoding Layer
            for q in range(self.n_qubits):
                qml.RY(inputs[q], wires=q)
                qml.RZ(inputs[q + self.n_qubits], wires=q)

            # Variational Layer mit Entanglement
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # --- 3. NOISE-HANDLING ---
        noise_model = build_noise_model(
            self.noise_type,
            self.p,
            self.noise_mode,
            self.two_qubit_factor
        )

        base_qnode = qml.QNode(circuit, self.dev, interface="torch")

        if noise_model is not None:
            self.qnode = qml.add_noise(base_qnode, noise_model)
        else:
            self.qnode = base_qnode

        # --- 4. TORCH LAYER ---
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier-Initialisierung mit reduziertem Gain für Stabilität."""
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        """
        Forward Pass durch den Quantenschaltkreis.

        Args:
            x: Input-Tensor (batch_size, num_features) oder (num_features,)

        Returns:
            Predictions im Bereich [0, 1] mit Shape (batch_size, 1)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Batch-Processing mit Fallback
        try:
            q_out = self.qlayer(x)
        except Exception:
            # Einzeln verarbeiten falls Batch fehlschlägt
            q_out = torch.stack([self.qlayer(sample) for sample in x])

        # Aggregation über Qubits und Normalisierung
        q_out_mean = torch.mean(q_out, dim=1)  # (batch_size, n_qubits) → (batch_size,)
        return ((q_out_mean + 1) / 2).view(-1, 1)  # [-1,1] → [0,1]