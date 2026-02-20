import json
import numpy as np
import torch
import gc

class QuantumJSONEncoder(json.JSONEncoder):
    """Zentrale Klasse für saubere JSON-Exporte im gesamten Projekt."""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

def cleanup_memory():
    """Zentrale Bereinigung für alle Module."""
    gc.collect()