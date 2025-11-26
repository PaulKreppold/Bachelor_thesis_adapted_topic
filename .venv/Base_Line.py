import torch
import torch.nn as nn
import pennylane as qml


class QuantumModel(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super(QuantumModel, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # StronglyEntanglingLayers (3 Parameter pro Qubit)
        self.weight_shapes = {"params": (self.num_layers, self.num_qubits, 3)}

        # WICHTIG: KEIN nn.Linear Layer!
        # Wir nutzen stattdessen einen festen Skalierungsfaktor für bessere Konvergenz.
        # Das ist kein "Layer", sondern ein Hyperparameter wie die Lernrate.
        self.scale = 5.0

        self.qnode = self.create_qnode()
        self.quantum_layer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes,
                                                init_method=lambda w: torch.nn.init.normal_(w, mean=0.0, std=0.05))

    def create_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            qml.templates.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True, pad_with=0.0)

            qml.templates.StronglyEntanglingLayers(params, wires=range(self.num_qubits))

            # WICHTIG: Wir messen nur Qubit 0 und 1
            # Qubit 0 = Score für Klasse 0
            # Qubit 1 = Score für Klasse 1
            # Die anderen Qubits (2-9) sind "Hidden Units" für die Berechnung
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        return circuit

    def forward(self, inputs):
        # Shape: [batch_size, 2]
        # Wertebereich: [-1, 1]
        q_out = self.quantum_layer(inputs)

        # Falls Batch Size 1 ist, Dimension korrigieren
        if len(q_out.shape) == 1:
            q_out = q_out.view(1, -1)

        # Skalierung (nötig für Softmax-Sättigung, aber KEIN lernbarer Parameter)
        # Damit werden aus [-1, 1] Werte wie [-5, 5], was "starken" Logits entspricht.
        logits = q_out * self.scale

        return logits