import config
from Base_Line import QuantumModel


def get_global_model(layers):
    """Erzeugt das ideale Referenzmodell für den Server (immer noiseless)."""
    return QuantumModel(layers, noise_type=None, p=0.0)


def get_client_model(sid, task_config):
    """
    Erzeugt das spezifische Modell für einen Client.

    Args:
        sid: Client-ID (z.B. "client_1" oder "client_1_sub_1")
        task_config: Szenario-Konfiguration mit noise_type, p, mapping, etc.

    Returns:
        QuantumModel mit korrekter Noise-Konfiguration

    Noise-Logik:
        - Bei type='hetero': Verwendet mapping[sid] oder mapping[base_client]
        - Undefinierte Clients in hetero-Szenarien → noiseless (None, 0.0)
        - Bei type='iso': Verwendet noise_type/p aus task_config
    """
    layers = task_config.get('layers', config.DEFAULT_LAYERS)
    n_mode = task_config.get('noise_mode', 'uniform')
    q2_factor = task_config.get('two_qubit_factor', config.TWO_QUBIT_NOISE_FACTOR)

    # Default: Noise aus task_config (für iso-Szenarien)
    n_t = task_config.get('noise_type', None)
    n_p = task_config.get('p', 0.0)

    # Für hetero-Szenarien: Client-spezifisches Mapping
    if task_config.get('type') == 'hetero':
        mapping = task_config.get('mapping', {})

        # Extrahiert Base-Client (client_1_sub_2 → client_1)
        parts = sid.split("_")
        base_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else sid

        # Lookup-Hierarchie: sid → base_id → default (noiseless)
        n_t, n_p = mapping.get(sid, mapping.get(base_id, (None, 0.0)))

    return QuantumModel(layers, n_t, n_p, noise_mode=n_mode, two_qubit_factor=q2_factor)