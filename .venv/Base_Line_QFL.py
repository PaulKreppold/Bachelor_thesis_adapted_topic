import torch
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import functools
import json
import pickle

# Eigene Module
import config
import data_prep
import plots
import aggregation
from Base_Line import QuantumModel
from train_and_eval import train_local_model, evaluate_model

# Mapping für bessere Lesbarkeit in CSV/Plots
STRAT_MAPPING = {
    "central": "Zentrale Baseline",
    "local": "Lokale Baseline",
    "qfl_global": "QFL Global",
    "qfl_local": "QFL Local (Personalisiert)"
}


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def convert_for_json(data):
    """Konvertiert NumPy-Typen rekursiv in JSON-kompatible Python-Typen."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_for_json(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    else:
        return data


# =============================================================================
# CORE EXPERIMENT LOGIC (SINGLE SEED)
# =============================================================================

def run_single_seed(seed, label, actual_type, n_p, rsna_root, chexpert_root, is_first_run=False):
    print(f"\n>>> [SEED {seed}] Starte Szenario: {label}")

    # Reproduzierbarkeit
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Daten laden
    loaders, meta = data_prep.get_federated_pca_loaders(
        rsna_root, chexpert_root, current_seed=seed
    )

    if is_first_run:
        data_prep.print_dataset_stats(loaders)
        data_prep.verify_disjoint_subclients(loaders)

    g_val_loader = aggregation.build_global_val_loader(loaders)
    cent_loader = data_prep.get_centralized_loader(loaders, config.BATCH_SIZE)

    model_dir = os.path.join(config.BASE_RESULTS_DIR, label, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Speicherstruktur initialisieren
    seed_data = {
        "metadata": {"seed": seed, "scenario": label, "noise_type": str(actual_type), "noise_p": n_p},
        "training": {
            "central": None,
            "local": {},
            "qfl_local_history": {},
            "qfl_global_rounds": []
        },
        "evaluation": {"central": {}, "local": {}, "qfl_local": {}, "qfl_global": {}},
        "confusion_matrices": {}
    }

    # ---------------------------------------------------------
    # 1. ZENTRALE BASELINE
    # ---------------------------------------------------------
    print(f"[{seed}] Training Zentrale Baseline...")
    central_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
    _, central_hist = train_local_model(
        central_model, cent_loader, g_val_loader,
        torch.optim.Adam(central_model.parameters(), lr=config.LR),
        config.EPOCHS_BASELINE, config.DEVICE, client_id=f"S{seed}_Central"
    )
    seed_data["training"]["central"] = central_hist
    torch.save(central_model.state_dict(), os.path.join(model_dir, f"S{seed}_Central.pth"))

    # Effiziente Eval: Einmal pro Base-Client
    for base_client in config.BASE_CLIENTS:
        rep_sub = f"{base_client}_sub_1"
        m_rep = evaluate_model(central_model, loaders[rep_sub]['test'], config.DEVICE, meta[rep_sub]['minority_idx'])
        seed_data["confusion_matrices"][(base_client, 'Zentrale')] = m_rep['cm']

        for i in range(1, config.NUM_SUBCLIENTS_PER_BASE + 1):
            sid = f"{base_client}_sub_{i}"
            seed_data["evaluation"]["central"][sid] = {
                **{k: v for k, v in m_rep.items() if k != 'cm'},
                "lds_ratio": meta[sid]["lds_ratio"]
            }

    # ---------------------------------------------------------
    # 2. LOKALE BASELINES
    # ---------------------------------------------------------
    print(f"[{seed}] Training Lokale Baselines...")
    for sid in config.ALL_SUBCLIENTS:
        local_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
        _, h = train_local_model(
            local_model, loaders[sid]['train'], loaders[sid]['val'],
            torch.optim.Adam(local_model.parameters(), lr=config.LR),
            config.EPOCHS_BASELINE, config.DEVICE, client_id=f"S{seed}_{sid}_Local"
        )
        seed_data["training"]["local"][sid] = h
        torch.save(local_model.state_dict(), os.path.join(model_dir, f"S{seed}_{sid}_Local.pth"))

        m = evaluate_model(local_model, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx'])
        seed_data["evaluation"]["local"][sid] = {
            **{k: v for k, v in m.items() if k != 'cm'},
            "lds_ratio": meta[sid]["lds_ratio"]
        }
        seed_data["confusion_matrices"][(sid, 'Lokal')] = m['cm']

    # ---------------------------------------------------------
    # 3. QUANTUM FEDERATED LEARNING (QFL)
    # ---------------------------------------------------------
    print(f"[{seed}] Starte QFL...")
    global_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)

    # History-Listen initialisieren (Inkl. global_step für Linearisierung)
    for sid in config.ALL_SUBCLIENTS:
        seed_data["training"]["qfl_local_history"][sid] = {
            "round_idx": [], "epoch_in_round": [], "global_step": [],
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

    final_local_models = {}

    for r in range(config.QFL_GLOBAL_ROUNDS):
        weights = []
        current_locals = {}
        # Offset berechnen: Wieviele Epochen wurden in vorigen Runden bereits trainiert?
        step_offset = r * config.QFL_LOCAL_EPOCHS

        for sid in config.ALL_SUBCLIENTS:
            loc_m = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
            loc_m.load_state_dict(global_model.state_dict())

            w, h = train_local_model(
                loc_m, loaders[sid]['train'], loaders[sid]['val'],
                torch.optim.Adam(loc_m.parameters(), lr=config.LR),
                config.QFL_LOCAL_EPOCHS, config.DEVICE, client_id=f"S{seed}_{sid}_R{r}"
            )
            weights.append(w)
            current_locals[sid] = loc_m

            # Linearisierte History Speicherung
            for e_idx in range(len(h["train_loss"])):
                hist = seed_data["training"]["qfl_local_history"][sid]
                hist["round_idx"].append(r)
                hist["epoch_in_round"].append(e_idx)
                hist["global_step"].append(step_offset + e_idx)  # Fortschreibung der X-Achse
                hist["train_loss"].append(h["train_loss"][e_idx])
                hist["train_acc"].append(h["train_acc"][e_idx])
                hist["val_loss"].append(h["val_loss"][e_idx])
                hist["val_acc"].append(h["val_acc"][e_idx])

        # Aggregation via FedAvg
        global_model.load_state_dict(aggregation.federated_averaging(weights))

        # Runden-Validierung des aggregierten Modells
        v_global = evaluate_model(global_model, g_val_loader, config.DEVICE)
        seed_data["training"]["qfl_global_rounds"].append(v_global)

        if r == config.QFL_GLOBAL_ROUNDS - 1:
            final_local_models = current_locals

    # Final QFL Evaluation
    torch.save(global_model.state_dict(), os.path.join(model_dir, f"S{seed}_QFL_Global_Final.pth"))

    for base_client in config.BASE_CLIENTS:
        rep_sub = f"{base_client}_sub_1"
        m_glob = evaluate_model(global_model, loaders[rep_sub]['test'], config.DEVICE, meta[rep_sub]['minority_idx'])
        seed_data["confusion_matrices"][(base_client, 'QFL_Global')] = m_glob['cm']

        for i in range(1, config.NUM_SUBCLIENTS_PER_BASE + 1):
            sid = f"{base_client}_sub_{i}"
            # QFL Global Metriken (Wissenstransfer-Check)
            seed_data["evaluation"]["qfl_global"][sid] = {
                **{k: v for k, v in m_glob.items() if k != 'cm'},
                "lds_ratio": meta[sid]["lds_ratio"]
            }
            # QFL Local Metriken (Personalisierungs-Check)
            loc_m = final_local_models[sid]
            m_loc = evaluate_model(loc_m, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx'])
            seed_data["evaluation"]["qfl_local"][sid] = {
                **{k: v for k, v in m_loc.items() if k != 'cm'},
                "lds_ratio": meta[sid]["lds_ratio"]
            }
            torch.save(loc_m.state_dict(), os.path.join(model_dir, f"S{seed}_QFL_{sid}_Local_Final.pth"))

    # Daten-Persistenz pro Seed
    hist_path = os.path.join(config.BASE_RESULTS_DIR, label, f"S{seed}_full_data.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(seed_data, f)

    # JSON Export (Schlankes Monitoring ohne CMs)
    json_view = convert_for_json({k: v for k, v in seed_data.items() if k != "confusion_matrices"})
    with open(hist_path.replace(".pkl", ".json"), "w") as f:
        json.dump(json_view, f, indent=2)

    return seed_data


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    master_results_list = []

    for n_type, n_p in config.PHASE1_SCENARIOS:
        label = f"{n_type}_p{n_p}" if n_type != "None" else "Noiseless"
        actual_type = None if n_type == "None" else n_type
        scenario_dir = os.path.join(config.BASE_RESULTS_DIR, label)
        os.makedirs(scenario_dir, exist_ok=True)

        # Parallelisierung über die Seeds
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            worker = functools.partial(
                run_single_seed, label=label, actual_type=actual_type, n_p=n_p,
                rsna_root=config.RSNA_ROOT, chexpert_root=config.CHEXPERT_ROOT,
                is_first_run=(n_type == "None" and n_p == 0)
            )
            scenario_seeds_data = list(executor.map(worker, config.SEEDS))

        # Flachklopfen der Ergebnisse für den CSV-Export (Punkt 4)
        rows = []
        for s_data in scenario_seeds_data:
            for strat_key in ["central", "local", "qfl_global", "qfl_local"]:
                for sid, metrics in s_data["evaluation"][strat_key].items():
                    rows.append({
                        "Seed": s_data["metadata"]["seed"],
                        "Szenario": STRAT_MAPPING[strat_key],
                        "Noise_Label": label,
                        "Client": "_".join(sid.split("_")[:2]),
                        "Subclient": sid.split("_")[-1],
                        **metrics  # Entpackt Metriken wie Accuracy, F1 etc.
                    })

        df_scenario = pd.DataFrame(rows)
        df_scenario.to_csv(os.path.join(scenario_dir, f"Summary_{label}.csv"), index=False)
        master_results_list.append(df_scenario)

    # Globaler Master-CSV Export
    master_df = pd.concat(master_results_list, ignore_index=True)
    master_df.to_csv(os.path.join(config.BASE_RESULTS_DIR, "Master_Results.csv"), index=False)

    print("\n" + "=" * 50)
    print("✅ SIMULATION ERFOLGREICH BEENDET")
    print(f"Ergebnisse gespeichert in: {config.BASE_RESULTS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()