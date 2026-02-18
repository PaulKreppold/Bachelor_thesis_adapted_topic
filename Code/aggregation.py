import torch
from torch.utils.data import DataLoader
from dataset_class import PneumoniaDataset


def federated_averaging(state_dicts):
    """
    Berechnet den Durchschnitt der Modellgewichte (FedAvg).
    """
    avg_state = {}
    # Initialisierung mit den Schlüsseln des ersten Modells
    for key in state_dicts[0].keys():
        # Stapeln der Tensoren über alle Clients und Durchschnittsbildung
        avg_state[key] = torch.mean(
            torch.stack([sd[key].float() for sd in state_dicts]), dim=0
        )
    return avg_state


def build_global_val_loader(all_fed_loaders, batch_size=32):
    """
    Erstellt einen globalen Validierungs-Loader aus den disjunkten
    Validierungssets der Basis-Clients.
    """
    xs, ys = [], []

    # Da alle Subclients eines Basis-Clients (z.B. client_1_sub_1 bis _4)
    # dasselbe disjunkte Validierungsset nutzen, nehmen wir nur jeweils
    # das Set vom ersten Subclient jeder Gruppe.
    base_clients = ["client_1", "client_2", "client_3", "client_4"]

    for base_id in base_clients:
        target_sub = f"{base_id}_sub_1"
        if target_sub in all_fed_loaders:
            # Zugriff auf die Attribute der PneumoniaDataset-Klasse
            ds = all_fed_loaders[target_sub]['val'].dataset
            xs.append(ds.dataset)  # Features
            ys.append(ds.labels)  # Labels

    # Kombinieren der Tensoren
    if not xs:
        return None

    X_global = torch.cat(xs, dim=0)
    Y_global = torch.cat(ys, dim=0)

    # Erstellung eines neuen PneumoniaDataset-Objekts
    global_ds = PneumoniaDataset(X_global, Y_global)

    return DataLoader(
        global_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True  # Verhindert Batch-Dimension-Fehler im Quantum-Layer
    )
