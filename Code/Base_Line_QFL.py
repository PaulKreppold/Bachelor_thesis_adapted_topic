import os
import sys

# =============================================================================
# 1. THREADING-SCHUTZ (MUSS VOR ALLEN ANDEREN IMPORTS STEHEN)
# =============================================================================
# Zwingt NumPy, PyTorch & Co. auf einen Kern pro Worker (Essenziell für AWS m8g)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import multiprocessing
import boto3
import traceback
from datetime import datetime

# Eigene Module
import config
import data_prep
import aggregation
import model_factory
from train_and_eval import train_local_model, evaluate_model
from utils import QuantumJSONEncoder, cleanup_memory

# PyTorch Global auf 1 Thread einschränken
torch.set_num_threads(1)


# =============================================================================
# HILFSFUNKTIONEN FÜR SPEICHERUNG UND CLOUD-SYNC
# =============================================================================

def upload_to_s3(file_path, bucket_name):
    """Lädt Dateien hoch und nutzt den Sicherheits-Präfix."""
    if not getattr(config, 'IS_AWS_CLOUD', False):
        return
    if not file_path or not os.path.exists(file_path):
        return
    try:
        s3_client = boto3.client('s3')
        # Ermittelt den Namen der Datei (z.B. S42_full_results.json)
        rel_path = os.path.relpath(file_path, config.BASE_RESULTS_DIR)

        # HIER PASSIERT DIE MAGIE:
        # Wir setzen den neuen Pfad DAVOR.
        # Das Ergebnis ist: "final_results/REPRESENTATIVE_MATRIX_RUN/..."
        # statt "results/..."
        s3_key = f"{config.S3_PREFIX}{rel_path}"

        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Sync: {s3_key}")  # Kurze Bestätigung im Log
    except Exception as e:
        print(f"⚠️ S3 Error: {e}")


def save_and_sync(obj, path, mode="pickle"):
    """Speichert Daten lokal und synchronisiert sie mit S3."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if mode == "pickle":
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        elif mode == "json":
            # HIER wird der QuantumJSONEncoder als Übersetzer genutzt
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, cls=QuantumJSONEncoder, indent=4)
        elif mode == "torch":
            torch.save(obj, path)

        upload_to_s3(path, config.S3_BUCKET_NAME)
    except Exception as e:
        print(f"⚠️ Save Error bei {path}: {e}")


# =============================================================================
# CORE TASK LOGIK (EIN SEED / EIN SZENARIO)
# =============================================================================

def run_single_task(task_config, seed):
    """Führt Training und Evaluation für eine spezifische Seed-Szenario-Kombi aus."""
    pid = os.getpid()
    torch.set_num_threads(1)  # Schutz im Subprozess

    label = task_config['label']
    layers = task_config['layers']

    print(f"[{pid}] [START] {label} | Seed: {seed}")

    try:
        # 1. Reproduzierbarkeit
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 2. Ordnerstruktur
        scenario_dir = os.path.join(config.BASE_RESULTS_DIR, label)
        model_seed_dir = os.path.join(scenario_dir, "models", f"S{seed}")
        os.makedirs(model_seed_dir, exist_ok=True)

        # 3. Datenvorbereitung
        loaders, meta = data_prep.get_federated_pca_loaders(
            config.RSNA_ROOT, config.CHEXPERT_ROOT, current_seed=seed, verbose=False
        )
        g_val_loader = aggregation.build_global_val_loader(loaders, config.BATCH_SIZE)
        cent_loader = data_prep.get_centralized_loader(loaders, config.BATCH_SIZE)

        seed_data = {
            "metadata": {"seed": seed, "scenario": label, "config": task_config},
            "history": {"centralized": None, "local_baselines": {}, "qfl_global": [], "qfl_local_history": {}},
            "evaluation": {"centralized": {}, "local": {}, "qfl_global": {}, "qfl_local": {}}
        }

        # --- PHASE 1: CENTRALIZED BASELINE ---
        c_mod = model_factory.get_global_model(layers).to(config.DEVICE)
        c_opt = torch.optim.Adam(c_mod.parameters(), lr=config.LEARNING_RATE)

        _, c_hist = train_local_model(
            c_mod, cent_loader, g_val_loader, c_opt,
            config.BASELINE_EPOCHS, config.DEVICE, f"CENT_{seed}"
        )
        seed_data["history"]["centralized"] = c_hist
        save_and_sync(c_mod.state_dict(), os.path.join(model_seed_dir, "central_baseline.pt"), "torch")

        for sid in config.ALL_SUBCLIENTS:
            seed_data["evaluation"]["centralized"][sid] = evaluate_model(
                c_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )

        del c_mod, c_opt
        cleanup_memory()

        # --- PHASE 2: LOCAL SILOS ---
        for sid in config.ALL_SUBCLIENTS:
            l_mod = model_factory.get_client_model(sid, task_config).to(config.DEVICE)
            l_opt = torch.optim.Adam(l_mod.parameters(), lr=config.LEARNING_RATE)

            _, l_hist = train_local_model(
                l_mod, loaders[sid]['train'], loaders[sid]['val'], l_opt,
                config.BASELINE_EPOCHS, config.DEVICE, ""
            )
            seed_data["history"]["local_baselines"][sid] = l_hist
            save_and_sync(l_mod.state_dict(), os.path.join(model_seed_dir, f"local_baseline_{sid}.pt"), "torch")

            seed_data["evaluation"]["local"][sid] = evaluate_model(
                l_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )
            del l_mod, l_opt
            cleanup_memory()

        # --- PHASE 3: QUANTUM FEDERATED LEARNING ---
        glob_mod = model_factory.get_global_model(layers).to(config.DEVICE)

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

                if sid not in seed_data["history"]["qfl_local_history"]:
                    seed_data["history"]["qfl_local_history"][sid] = {
                        k: [] for k in h.keys() if k != 'convergence_metrics'
                    }
                for k in seed_data["history"]["qfl_local_history"][sid].keys():
                    seed_data["history"]["qfl_local_history"][sid][k].extend(h[k])

                if r == config.GLOBAL_ROUNDS - 1:
                    save_and_sync(cl_m.state_dict(), os.path.join(model_seed_dir, f"qfl_local_final_{sid}.pt"), "torch")
                    seed_data["evaluation"]["qfl_local"][sid] = evaluate_model(
                        cl_m, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
                    )
                del cl_m, cl_opt
                cleanup_memory()

            new_glob_w = aggregation.federated_averaging(round_weights)
            glob_mod.load_state_dict(new_glob_w)

            g_eval = evaluate_model(glob_mod, g_val_loader, config.DEVICE)
            seed_data["history"]["qfl_global"].append({
                "round": r,
                **{k.replace("test_", "val_"): v for k, v in g_eval.items()}
            })

        save_and_sync(glob_mod.state_dict(), os.path.join(model_seed_dir, "qfl_global_final.pt"), "torch")
        for sid in config.ALL_SUBCLIENTS:
            seed_data["evaluation"]["qfl_global"][sid] = evaluate_model(
                glob_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )

        # --- FINALE SPEICHERUNG (HIER SIND PKL UND JSON) ---
        save_and_sync(seed_data, os.path.join(scenario_dir, f"S{seed}_full_results.pkl"), "pickle")
        save_and_sync(seed_data, os.path.join(scenario_dir, f"S{seed}_full_results.json"), "json")

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
# MAIN SCHEDULER
# =============================================================================

def main():
    try:
        # Wichtig für CUDA/Multiprocessing Stabilität auf AWS
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- NEUE LOGIK: FILTERUNG NACH SPECIFIC SEEDS ---
    all_tasks = []
    for scenario in config.ALL_SCENARIOS:
        # Wir prüfen, ob für dieses Szenario explizit Seeds definiert wurden
        if 'specific_seeds' in scenario:
            for seed in scenario['specific_seeds']:
                all_tasks.append((scenario, seed))

    # Shuffle sorgt dafür, dass schwere und leichte Szenarien gemischt werden
    random.shuffle(all_tasks)

    master_rows = []

    print(f"\n🚀 QFL SIMULATION ENGINE GESTARTET")
    print(f"   Instanz:  {getattr(config, 'S3_PREFIX', 'UNKNOWN')}")
    print(f"   Worker:   {config.MAX_WORKERS}")
    print(f"   Tasks:    {len(all_tasks)}\n")

    # Ausführung der gefilterten Task-Liste
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_task, t, s) for t, s in all_tasks]

        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Simulation"):
            res = future.result()
            if res:
                # Metriken für die Master-CSV sammeln
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

                # Regelmäßiges Backup der Master-Liste nach S3
                if len(master_rows) > 0 and len(master_rows) % 100 == 0:
                    df_tmp = pd.DataFrame(master_rows)
                    out_tmp = os.path.join(config.BASE_RESULTS_DIR, "Master_Results_Partial.csv")
                    df_tmp.to_csv(out_tmp, index=False)
                    upload_to_s3(out_tmp, config.S3_BUCKET_NAME)

    # Finale Speicherung und Sync
    if master_rows:
        output_path = os.path.join(config.BASE_RESULTS_DIR, "Master_Results.csv")
        pd.DataFrame(master_rows).to_csv(output_path, index=False)
        upload_to_s3(output_path, config.S3_BUCKET_NAME)
        print(f"\n✅ Simulation beendet. Daten unter {config.S3_PREFIX} auf S3 gesichert.")


if __name__ == "__main__":
    main()