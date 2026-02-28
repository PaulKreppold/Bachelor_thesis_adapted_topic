import torch
import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
import random
from datetime import datetime
import gc

# Eigene Module
import config
import data_prep
import aggregation
import model_factory
# WICHTIG: Hier die neue Funktion importieren
from train_and_eval import train_local_model, evaluate_model, compute_unified_convergence_metrics
from utils import QuantumJSONEncoder


def test_setup_consistency():
    print("\n--- 🔍 STARTING PRE-FLIGHT CHECK ---")

    # 1. Teste Daten-Orchestrierung
    loaders, meta = data_prep.get_federated_pca_loaders(
        config.RSNA_ROOT, config.CHEXPERT_ROOT, current_seed=42, verbose=True
    )

    # 2. Teste Mapping-Logik für jedes Szenario
    for scenario in config.ALL_SCENARIOS:
        print(f"\nScenario: {scenario['label']}")
        for sid in config.ALL_SUBCLIENTS:
            # Simuliere Modell-Erstellung
            test_model = model_factory.get_client_model(sid, scenario)

            # Prüfe, ob Rauschen korrekt gesetzt wurde
            n_type = test_model.noise_type
            n_p = test_model.p

            # Logik-Check: Erhält der Subclient das Rauschen seiner Base-Klinik?
            print(f"  ├─ {sid}: Noise={n_type}, p={n_p} | LDS-Idx={meta[sid]['minority_idx']}")

            # Kleiner Assert zur Sicherheit
            if scenario['group'] == "bias_core" and "client_1" in sid:
                assert n_p > 0 or n_type is not None, f"FEHLER: {sid} sollte Rauschen haben!"

    print("\n✅ Pre-Flight Check erfolgreich: Alle Mappings und Daten-Silos sind konsistent.")

def run_single_task(task_config, seed):
    """
    Führt ein Experiment aus mit Fokus auf Speichereffizienz und Datenkonsistenz.
    """
    log_path = os.path.join(config.BASE_RESULTS_DIR, "simulation.log")
    original_stdout = sys.stdout
    f_log = open(log_path, "a", encoding="utf-8")

    # Lokale Variablen initialisieren für sicheren Cleanup
    loaders = None
    c_mod = l_mod = glob_mod = None

    try:
        sys.stdout = f_log
        label = task_config['label']
        layers = task_config['layers']

        # Reproduzierbarkeit der Modell-Initialisierung (innerhalb des Tasks)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if config.DEVICE.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        scenario_dir = os.path.join(config.BASE_RESULTS_DIR, label)
        model_dir = os.path.join(scenario_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{'=' * 80}\n[START] Scenario: {label} | Seed: {seed}\n{'=' * 80}")

        # 1. DATEN LADEN (PCA nutzt config.CALIBRATION_SEED intern)
        loaders, meta = data_prep.get_federated_pca_loaders(
            config.RSNA_ROOT, config.CHEXPERT_ROOT, current_seed=seed, verbose=False
        )
        g_val_loader = aggregation.build_global_val_loader(loaders, batch_size=config.BATCH_SIZE)
        cent_loader = data_prep.get_centralized_loader(loaders, config.BATCH_SIZE)

        seed_data = {
            "metadata": {"seed": seed, "scenario": label, "config": task_config},
            "history": {"centralized": None, "local_baselines": {}, "qfl_global": [], "qfl_local_history": {}},
            "evaluation": {"centralized": {}, "local": {}, "qfl_global": {}, "qfl_local": {}}
        }

        # --- PHASE 1: CENTRAL (Upper Bound) ---
        c_mod = model_factory.get_global_model(layers).to(config.DEVICE)
        _, c_hist = train_local_model(
            c_mod, cent_loader, g_val_loader,
            torch.optim.Adam(c_mod.parameters(), lr=config.LEARNING_RATE),
            config.BASELINE_EPOCHS, config.DEVICE, f"{label}_S{seed}_CENTRAL"
        )
        # NEU: Unified Metrics für Centralized
        c_hist['convergence_metrics'] = compute_unified_convergence_metrics(c_hist, mode="central")
        seed_data["history"]["centralized"] = c_hist
        torch.save(c_mod.state_dict(), os.path.join(model_dir, f"central_S{seed}.pth"))

        for sid in config.ALL_SUBCLIENTS:
            e = evaluate_model(c_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx'])
            e['convergence_metrics'] = c_hist['convergence_metrics']
            seed_data["evaluation"]["centralized"][sid] = e

        del c_mod

        # --- PHASE 2: LOCAL SILO (Baselines) ---
        for sid in config.ALL_SUBCLIENTS:
            l_mod = model_factory.get_client_model(sid, task_config).to(config.DEVICE)
            _, l_hist = train_local_model(
                l_mod, loaders[sid]['train'], loaders[sid]['val'],
                torch.optim.Adam(l_mod.parameters(), lr=config.LEARNING_RATE),
                config.BASELINE_EPOCHS, config.DEVICE, f"{label}_S{seed}_LOCAL_{sid}"
            )
            # NEU: Unified Metrics für Local
            l_hist['convergence_metrics'] = compute_unified_convergence_metrics(l_hist, mode="local")
            seed_data["history"]["local_baselines"][sid] = l_hist
            torch.save(l_mod.state_dict(), os.path.join(model_dir, f"local_silo_{sid}_S{seed}.pth"))

            e = evaluate_model(l_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx'])
            e['convergence_metrics'] = l_hist['convergence_metrics']
            seed_data["evaluation"]["local"][sid] = e
            del l_mod

        # --- PHASE 3: QFL (Federated Learning) ---
        glob_mod = model_factory.get_global_model(layers).to(config.DEVICE)

        for r in range(config.GLOBAL_ROUNDS):
            round_weights = []
            step_offset = r * config.LOCAL_EPOCHS

            for sid in config.ALL_SUBCLIENTS:
                cl_m = model_factory.get_client_model(sid, task_config).to(config.DEVICE)
                cl_m.load_state_dict(glob_mod.state_dict())

                w, h = train_local_model(
                    cl_m, loaders[sid]['train'], loaders[sid]['val'],
                    torch.optim.Adam(cl_m.parameters(), lr=config.LEARNING_RATE),
                    config.LOCAL_EPOCHS, config.DEVICE, f"{label}_S{seed}_QFL_R{r}_{sid}"
                )
                round_weights.append(w)

                if sid not in seed_data["history"]["qfl_local_history"]:
                    seed_data["history"]["qfl_local_history"][sid] = {
                        "train_acc": [], "val_acc": [], "train_loss": [],
                        "val_loss": [], "global_step": [], "epoch_times": []
                    }

                hr = seed_data["history"]["qfl_local_history"][sid]
                for idx in range(len(h['train_acc'])):
                    hr["train_acc"].append(h['train_acc'][idx])
                    hr["val_acc"].append(h['val_acc'][idx])
                    hr["train_loss"].append(h['train_loss'][idx])
                    hr["val_loss"].append(h['val_loss'][idx])
                    hr["global_step"].append(step_offset + idx)
                    hr["epoch_times"].append(h["epoch_times"][idx])

                if r == config.GLOBAL_ROUNDS - 1:
                    torch.save(cl_m.state_dict(), os.path.join(model_dir, f"qfl_local_final_{sid}_S{seed}.pth"))
                    eval_metrics = evaluate_model(cl_m, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx'])
                    # NEU: Unified Metrics für Federated (isoliert Runden-Enden)
                    eval_metrics['convergence_metrics'] = compute_unified_convergence_metrics(
                        hr, mode="federated", local_epochs=config.LOCAL_EPOCHS
                    )
                    seed_data["evaluation"]["qfl_local"][sid] = eval_metrics

                del cl_m

            new_global_weights = aggregation.federated_averaging(round_weights)
            glob_mod.load_state_dict(new_global_weights)

            raw_global_metrics = evaluate_model(glob_mod, g_val_loader, config.DEVICE)
            clean_global_metrics = {"round": r,
                                    **{k.replace("test_", "val_"): v for k, v in raw_global_metrics.items()}}
            seed_data["history"]["qfl_global"].append(clean_global_metrics)

        torch.save(glob_mod.state_dict(), os.path.join(model_dir, f"qfl_global_final_S{seed}.pth"))

        # Finale Evaluation des globalen Modells
        for sid in config.ALL_SUBCLIENTS:
            seed_data["evaluation"]["qfl_global"][sid] = evaluate_model(
                glob_mod, loaders[sid]['test'], config.DEVICE, meta[sid]['minority_idx']
            )

        with open(os.path.join(scenario_dir, f"S{seed}_full_results.pkl"), "wb") as pf:
            pickle.dump(seed_data, pf)
        with open(os.path.join(scenario_dir, f"S{seed}_metrics.json"), "w") as jf:
            json.dump(seed_data, jf, indent=4, cls=QuantumJSONEncoder)

        print(f"[SUCCESS] Scenario {label} | Seed {seed} completed.")
        return seed_data

    except Exception as err:
        print(f"\n{'!' * 80}\n[ERROR] Scenario {task_config.get('label', 'UNKNOWN')}, Seed {seed}: {str(err)}")
        print(traceback.format_exc())
        return None

    finally:
        # === SCHRITT 1: EXPLIZITE ZUWEISUNG (Laut Review sicherste Methode) ===
        loaders = cent_loader = g_val_loader = None
        c_mod = l_mod = glob_mod = None

        # === SCHRITT 2: CPU-CLEANUP ===
        gc.collect()

        # === SCHRITT 3: GPU-CLEANUP (VRAM) ===
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # === SCHRITT 4: LOGGING ===
        try:
            now_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{now_str}] 🧹 Cleanup completed. RAM/VRAM resources released.")
        except:
            pass

        sys.stdout = original_stdout
        f_log.close()

def main():
    # --- NEU: PRE-FLIGHT CHECK ---
    # Führt eine Trockenübung für Daten und Mappings durch
    try:
        test_setup_consistency()
    except Exception as e:
        print(f"❌ PRE-FLIGHT CHECK FEHLGESCHLAGEN: {e}")
        return  # Beendet das Programm vor dem Start der Tasks

    print("🔍 Initialisiere Datensätze und starte PCA-Orchestrierung...")
    # Initialer Check nutzt CALIBRATION_SEED intern
    data_prep.get_federated_pca_loaders(config.RSNA_ROOT, config.CHEXPERT_ROOT, verbose=True)

    all_tasks = [(s, seed) for s in config.ALL_SCENARIOS for seed in config.RANDOM_SEEDS]
    master_rows = []
    log_file = os.path.join(config.BASE_RESULTS_DIR, "simulation.log")

    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"🚀 Starte {len(all_tasks)} parallele Aufgaben auf {config.MAX_WORKERS} Workern.")

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_task, t, s) for t, s in all_tasks]

        with tqdm(total=len(all_tasks), desc="Simulations-Fortschritt", unit="task") as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        for strategy in ["centralized", "local", "qfl_local", "qfl_global"]:
                            for sid, metrics in res["evaluation"][strategy].items():
                                row = {
                                    "seed": res["metadata"]["seed"],
                                    "scenario": res["metadata"]["scenario"],
                                    "group": res["metadata"]["config"]["group"],
                                    "strategy": strategy,
                                    "client": sid,
                                }
                                for m_name, m_val in metrics.items():
                                    if m_name == 'convergence_metrics':
                                        for c_name, c_val in m_val.items():
                                            row[f"conv_{c_name}"] = c_val
                                    elif m_name != 'confusion_matrix':
                                        row[m_name] = m_val
                                master_rows.append(row)
                except Exception as e:
                    print(f"Kritischer Task-Fehler: {e}")
                pbar.update(1)

    if master_rows:
        df_master = pd.DataFrame(master_rows)
        output_path = os.path.join(config.BASE_RESULTS_DIR, "Master_Results.csv")
        df_master.to_csv(output_path, index=False)
        print(f"\n✅ Simulation beendet. Ergebnisse gespeichert in: {output_path}")
    else:
        print("\n❌ Keine Daten für Master_Results.csv generiert.")

if __name__ == "__main__":
    main()