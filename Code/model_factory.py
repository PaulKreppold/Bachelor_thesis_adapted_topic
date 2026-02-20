# Wir importieren NUR das Modell, keine globale Config
from Base_Line import QuantumModel

def get_global_model(task_config):
    """
    Erzeugt das Modell basierend auf dem übergebenen Szenario-Paket.
    Verlässt sich zu 100% auf die Keys aus der ALL_SCENARIOS-Schleife.
    """
    return QuantumModel(
        num_qubits = task_config['num_qubits'],
        num_layers = task_config['num_layers'],
        ansatz     = task_config['ansatz']
    )

def get_client_model(sid, task_config):
    """
    In der noiseless Ablation identisch zum globalen Modell.
    'sid' wird hier ignoriert, da alle Clients die gleiche Architektur erhalten.
    """
    return get_global_model(task_config)