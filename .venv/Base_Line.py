import torch
import torch.nn as nn
import pennylane as qml


class QuantumModelDRU(nn.Module):
    def __init__(self, num_qubits=6, num_features=48, layers_per_block=1):
        """
        Reiner Variational Quantum Classifier (VQC) mit Data Re-Uploading.
        Ohne klassische Skalierungsparameter.

        Args:
            num_qubits (int): Anzahl der Qubits (6).
            num_features (int): Gesamteingabe (48 PCA-Komponenten).
            layers_per_block (int): Anzahl der trainierbaren Schichten nach jedem Daten-Upload.
        """
        super().__init__()
        self.n_qubits = num_qubits
        self.n_features = num_features
        # 12 Features pro Block (6x RY, 6x RZ)
        self.features_per_block = self.n_qubits * 2
        self.n_blocks = self.n_features // self.features_per_block

        # Device Definition
        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Gewichte-Shape: (Blöcke, Schichten, Qubits, 3 Rotationswinkel)
        weight_shapes = {"weights": (self.n_blocks, layers_per_block, self.n_qubits, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(self.n_blocks):
                # --- ENCODING (Re-Uploading) ---
                start_idx = i * self.features_per_block
                block_input = inputs[:, start_idx: start_idx + self.features_per_block]

                for q in range(self.n_qubits):
                    # Nutzt RY und RZ für kompaktes Encoding der 12 Features pro Block
                    qml.RY(block_input[:, q], wires=q)
                    qml.RZ(block_input[:, q + self.n_qubits], wires=q)

                # --- VARIATIONAL LAYER ---
                qml.StronglyEntanglingLayers(weights[i], wires=range(self.n_qubits))

            # Rückgabe des Erwartungswerts zwischen -1 und 1
            return qml.expval(qml.PauliZ(0))

        # PennyLane QNN Integration
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Initialisierung der Quanten-Parameter
        with torch.no_grad():
            nn.init.xavier_normal_(self.qlayer.weights, gain=0.1)

    def forward(self, x):
        # Quanten-Output (-1 bis 1)
        q_out = self.qlayer(x).reshape(-1, 1)

        # Mapping auf (0 bis 1), um es als Wahrscheinlichkeit zu interpretieren
        # Das ist "purer" als ein klassisches Scaling, da es nur den Wertebereich verschiebt
        return (q_out + 1) / 2
