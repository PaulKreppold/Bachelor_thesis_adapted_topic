import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import random
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Eigene Module
import config
import data_prep
import aggregation
import model_factory
from train_and_eval import train_local_model, evaluate_model
from utils import QuantumJSONEncoder, cleanup_memory

# =============================================================================
# 1. THREADING-OPTIMIERUNG (Für parallele Simulation am Mac)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)


# =============================================================================
# 2. HILFSFUNKTIONEN FÜR SPEICHERUNG
# =============================================================================

def save_results(obj, path, mode="json"):
    """Sichert Metriken und Modelle konsistent zum Hauptprojekt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if mode == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, cls=QuantumJSONEncoder, indent=4)
        elif mode == "pickle":
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        elif mode == "torch":
            torch.save(obj, path)
    except Exception as e:
        print(f"⚠️ Speicherfehler unter {path}: {e}")


# =============================================================================
# 3. CORE TASK LOGIK (EIN SEED / EIN SZENARIO)
# =============================================================================

def run_single_task(task_config, seed):
    """
    Führt die komplette Simulation (Zentral, Lokal, QFL) für AE aus.
    Struktur ist zu 100% identisch mit dem PCA-Lauf für volle Vergleichbarkeit.
    """
    pid = os.getpid()
    label = task_config['label']
    torch.set_num_threads(1)

    # Reproduzierbarkeit
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ordnerstruktur
    scenario_dir = os.path.join(config.BASE_RESULTS_DIR, label)
    model_seed_dir = os.path.join(scenario_dir, "models", f"S{seed}")
    os.makedirs(model_seed_dir, exist_ok=True)

    try:
        # --- PHASE 1: DATEN VORBEREITEN ---
        loaders, meta = data_prep.get_federated_amplitude_loaders(
            config.RSNA_ROOT,
            config.CHEXPERT_ROOT,
            current_seed=seed,
            num_qubits=task_config['num_qubits']
        )
        g_val_loader = aggregation.build_global_val_loader(loaders, config.BATCH_SIZE)
        cent_loader = data_prep.get_centralized_loader(loaders, config.BATCH_SIZE)

        seed_data = {
            "metadata": {"seed": seed, "scenario": label, "config": task_config},
            "history": {
                "centralized": None,
                "local_baselines": {},
                "qfl_global": [],
                "qfl_local_history": {}
            },
            "evaluation": {
                "centralized": {},
                "local": {},
                "qfl_global": {},
                "qfl_local": {}
            }
        }

        # --- PHASE 2: CENTRALIZED BASELINE (Trainiert auf allen 16 Clients) ---
        c_mod = model_factory.get_global_model(task_config).to(config.DEVICE)
        c_opt = torch.optim.Adam(c_mod.parameters(), lr=config.LEARNING_RATE)

        _, c_hist = train_local_model(
            c_mod, cent_loader, g_val_loader, c_opt,
            config.BASELINE_EPOCHS, config.DEVICE, f"CENT_{seed}"
        )
        seed_data["history"]["centralized"] = c_hist
        save_results(c_mod.state_dict(), os.path.join(model_seed_dir, "central_baseline.pt"), "torch")

        for sid in config.ALL_SUBCLIENTS:
            seed_data["evaluation"]["centralized"][sid] = evaluate_model(
                c_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )

        del c_mod, c_opt
        cleanup_memory()

        # --- PHASE 3: LOCAL SILOS (Alle 16 Subclients einzeln) ---
        for sid in config.ALL_SUBCLIENTS:
            l_mod = model_factory.get_client_model(sid, task_config).to(config.DEVICE)
            l_opt = torch.optim.Adam(l_mod.parameters(), lr=config.LEARNING_RATE)

            _, l_hist = train_local_model(
                l_mod, loaders[sid]['train'], loaders[sid]['val'], l_opt,
                config.BASELINE_EPOCHS, config.DEVICE, ""
            )
            seed_data["history"]["local_baselines"][sid] = l_hist
            save_results(l_mod.state_dict(), os.path.join(model_seed_dir, f"local_baseline_{sid}.pt"), "torch")

            seed_data["evaluation"]["local"][sid] = evaluate_model(
                l_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )
            del l_mod, l_opt
            cleanup_memory()

        # --- PHASE 4: QUANTUM FEDERATED LEARNING (QFL) ---
        glob_mod = model_factory.get_global_model(task_config).to(config.DEVICE)

        for r in range(config.GLOBAL_ROUNDS):
            round_weights = []
            for sid in config.ALL_SUBCLIENTS:
                cl_m = model_factory.get_client_model(sid, task_config).to(config.DEVICE)
                cl_m.load_state_dict(glob_mod.state_dict())
                cl_opt = torch.optim.Adam(cl_m.parameters(), lr=config.LEARNING_RATE)

                w, h = train_local_model(
                    cl_m, loaders[sid]['train'], loaders[sid]['val'], cl_opt,
                    config.LOCAL_EPOCHS, config.DEVICE, ""
                )
                round_weights.append(w)

                # History pro Client tracken (identisch zu PCA)
                if sid not in seed_data["history"]["qfl_local_history"]:
                    seed_data["history"]["qfl_local_history"][sid] = {
                        k: [] for k in h.keys() if k != 'convergence_metrics'
                    }
                for k in seed_data["history"]["qfl_local_history"][sid].keys():
                    seed_data["history"]["qfl_local_history"][sid][k].extend(h[k])

                if r == config.GLOBAL_ROUNDS - 1:
                    save_results(cl_m.state_dict(), os.path.join(model_seed_dir, f"qfl_local_final_{sid}.pt"), "torch")
                    seed_data["evaluation"]["qfl_local"][sid] = evaluate_model(
                        cl_m, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
                    )
                del cl_m, cl_opt
                cleanup_memory()

            # Aggregation via FedAvg
            new_glob_w = aggregation.federated_averaging(round_weights)
            glob_mod.load_state_dict(new_glob_w)

            # Globales Validierungs-Monitoring
            g_eval = evaluate_model(glob_mod, g_val_loader, config.DEVICE)
            seed_data["history"]["qfl_global"].append({
                "round": r,
                **{k.replace("test_", "val_"): v for k, v in g_eval.items()}
            })

        save_results(glob_mod.state_dict(), os.path.join(model_seed_dir, "qfl_global_final.pt"), "torch")
        for sid in config.ALL_SUBCLIENTS:
            seed_data["evaluation"]["qfl_global"][sid] = evaluate_model(
                glob_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )

        # Finale Speicherung der Metriken
        save_results(seed_data, os.path.join(scenario_dir, f"S{seed}_full_results.pkl"), "pickle")
        save_results(seed_data, os.path.join(scenario_dir, f"S{seed}_full_results.json"), "json")

        del glob_mod
        cleanup_memory()

        print(f"[{pid}] [SUCCESS] {label} | Seed {seed}")
        return seed_data

    except Exception as err:
        print(f"[{pid}] [ERROR] {label} (Seed {seed}): {err}")
        traceback.print_exc()
        return None
    finally:
        cleanup_memory()


# =============================================================================
# 4. MAIN SCHEDULER
# =============================================================================

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    all_tasks = []
    for scenario in config.ALL_SCENARIOS:
        if 'specific_seeds' in scenario:
            for seed in scenario['specific_seeds']:
                all_tasks.append((scenario, seed))

    random.shuffle(all_tasks)

    master_rows = []
    max_workers = config.MAX_WORKERS

    print(f"\n🚀 AE ABLATION ENGINE GESTARTET (Waterproof Comparison)")
    print(f"   Worker:  {max_workers}")
    print(f"   Tasks:   {len(all_tasks)}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_task, t, s) for t, s in all_tasks]

        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Ablation"):
            res = future.result()
            if res:
                for strategy in ["centralized", "local", "qfl_local", "qfl_global"]:
                    for sid, metrics in res["evaluation"][strategy].items():
                        row = {
                            "seed": res["metadata"]["seed"],
                            "scenario": res["metadata"]["scenario"],
                            "group": res["metadata"]["config"]["group"],
                            "strategy": strategy,
                            "client": sid
                        }
                        for m_name, m_val in metrics.items():
                            if isinstance(m_val, (int, float, str)):
                                row[m_name] = m_val
                        master_rows.append(row)

                # Backup Master CSV
                if len(master_rows) % 100 == 0:
                    pd.DataFrame(master_rows).to_csv(
                        os.path.join(config.BASE_RESULTS_DIR, "Master_Results_Ablation_Partial.csv"),
                        index=False
                    )

    if master_rows:
        output_path = os.path.join(config.BASE_RESULTS_DIR, "Master_Results_Ablation.csv")
        pd.DataFrame(master_rows).to_csv(output_path, index=False)
        print(f"\n✅ Ablation beendet. Ergebnisse unter: {config.BASE_RESULTS_DIR}")


if __name__ == "__main__":
    main()