import torch
import torch.nn as nn
import pennylane as qml

import torch
import torch.nn as nn
import pennylane as qml
import config


class QuantumModelDRU(nn.Module):
    def __init__(self, num_qubits=6, num_features=48, layers_per_block=1,
                 encoding_type='dense', circuit_type='strongly_entangling',
                 use_reuploading=True, trainable_enc=False, measurement='mean'):
        super().__init__()
        self.n_qubits = num_qubits
        self.n_features = num_features
        self.encoding_type = encoding_type
        self.circuit_type = circuit_type
        self.use_reuploading = use_reuploading
        self.trainable_enc = trainable_enc
        self.measurement = measurement

        self.feats_per_qubit = 2 if encoding_type == 'dense' else 1
        self.feats_per_block = self.n_qubits * self.feats_per_qubit
        self.n_blocks = max(1, self.n_features // self.feats_per_block)

        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Weight Shapes definieren
        weight_shapes = {}
        if circuit_type == 'strongly_entangling':
            weight_shapes["weights"] = (self.n_blocks, layers_per_block, self.n_qubits, 3)
        else:  # hardware_efficient
            weight_shapes["weights"] = (self.n_blocks, layers_per_block, self.n_qubits)

        if self.trainable_enc:
            weight_shapes["enc_w"] = (self.n_features,)
            weight_shapes["enc_b"] = (self.n_features,)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, **kwargs):
            # Feature Tuning: w*x + b
            if self.trainable_enc:
                x = inputs * kwargs["enc_w"] + kwargs["enc_b"]
            else:
                x = inputs

            if not self.use_reuploading:
                for b in range(self.n_blocks): self._apply_encoding(x, b)
                for b in range(self.n_blocks): self._apply_ansatz(weights[b])
            else:
                for b in range(self.n_blocks):
                    self._apply_encoding(x, b)
                    self._apply_ansatz(weights[b])

            if self.measurement == 'first':
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Initialisierung
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)
            if self.trainable_enc:
                nn.init.ones_(self.qlayer.enc_w)
                nn.init.zeros_(self.qlayer.enc_b)

    def _apply_encoding(self, inputs, block_idx):
        """Führt das gewählte Encoding für einen Block aus (Batch-sicher)."""
        start = block_idx * self.feats_per_block

        # Sicherheits-Check: Falls mehr Blöcke als Features vorhanden sind
        if start >= self.n_features:
            return

        for q in range(self.n_qubits):
            # KORREKTUR: Nutze [:, start + q], um das Feature über den gesamten Batch zu greifen
            qml.RY(inputs[:, start + q], wires=q)

            if self.encoding_type == 'dense':
                # Sicherstellen, dass der RZ-Index existiert
                rz_idx = start + q + self.n_qubits
                if rz_idx < self.n_features:
                    qml.RZ(inputs[:, rz_idx], wires=q)

    def _apply_ansatz(self, block_weights):
        if self.circuit_type == 'strongly_entangling':
            qml.StronglyEntanglingLayers(block_weights, wires=range(self.n_qubits))
        else:
            for layer_w in block_weights:
                for q in range(self.n_qubits): qml.RY(layer_w[q], wires=q)
                for q in range(self.n_qubits): qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

    def forward(self, x):
        q_out = self.qlayer(x)
        if self.measurement == 'mean' and q_out.dim() > 1:
            q_out = torch.mean(q_out, dim=1)
        return ((q_out + 1) / 2).reshape(-1, 1)