import pennylane as qml
import config


def build_noise_model(noise_type, p, noise_mode="uniform", two_qubit_factor=config.TWO_QUBIT_NOISE_FACTOR):
    if noise_type in [None, "None", "baseline"] or p == 0.0:
        return None

    noise_ops = {
        "bit_flip": qml.BitFlip,
        "phase_flip": qml.PhaseFlip,
        "depolarizing": qml.DepolarizingChannel,
        "amplitude_damping": qml.AmplitudeDamping,
        "phase_damping": qml.PhaseDamping,
    }

    NoiseOp = noise_ops[noise_type]
    single_qubit_gates = [qml.RY, qml.RZ, qml.Rot]
    two_qubit_gates = [qml.CNOT]

    # Noise-Funktion mit Default-Parameter
    # FIX 1: **kwargs in der Signatur hinzufügen
    def apply_noise_per_wire(op, p_val=p, **kwargs):
        for wire in op.wires:
            NoiseOp(p_val, wires=wire)

    noise_map = {}

    if noise_mode == "uniform":
        cond = qml.noise.op_in(single_qubit_gates + two_qubit_gates)
        noise_map[cond] = apply_noise_per_wire
    else:
        p_two_qubit = min(p * two_qubit_factor, 1.0)
        noise_map[qml.noise.op_in(single_qubit_gates)] = apply_noise_per_wire

        # FIX 2: Das Lambda muss ebenfalls **kwargs annehmen und weiterreichen
        noise_map[qml.noise.op_in(two_qubit_gates)] = lambda op, **kwargs: apply_noise_per_wire(op, p_val=p_two_qubit,
                                                                                                **kwargs)

    # Readout Noise
    noise_map[qml.noise.op_in([qml.measure])] = apply_noise_per_wire

    return qml.NoiseModel(noise_map)