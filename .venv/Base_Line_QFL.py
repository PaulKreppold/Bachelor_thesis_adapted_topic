import torch
import torch.nn as nn
import numpy as np
import random
import copy
import sys
import os
from collections import Counter
from torch.utils.data import ConcatDataset, DataLoader

from Base_Line import QuantumModel
from config import *
from data_prep import load_client_data, get_labels_from_dataset
from train_and_eval import train_client, evaluate_model
from aggregation_method import aggregate_models

from plots import (
    plot_per_client_boxplots, plot_qfl_vs_baseline_accuracy, plot_global_boxplot,
    plot_minority_f1_boxplots, plot_f1_vs_minority_ratio, plot_minority_f1_density,
    plot_stability_boxplot, plot_auc_heatmap_skewed, plot_auc_pr_heatmap_skewed,
    plot_confusion_matrix_skewed, plot_roc_pr_skewed, plot_roc_pr_comparison_all_clients,
    compute_bias_summary, identify_skewed_clients, analyze_lds_resilience,
    plot_skew_severity_analysis
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



seeds = [42, 1337]
num_rounds = 15
local_epochs = 10
learning_rate = 0.01
mu = 0.01  # FedProx Proximal Term (gegen Client Drift)

baseline_epochs = num_rounds * local_epochs

global_m_num_qubits = 10
global_m_num_layers = 4
client_m_num_layers = 4

print("\nLoading all client data...")

client_train_loaders, client_val_loaders, client_test_loaders, external_loader = load_client_data(
    client_qubits=client_qubits,
    custom_collate_fn=None,
    seed=42
)

active_clients = sorted(list(client_train_loaders.keys()))
if not active_clients:
    print("CRITICAL ERROR: Keine Clients gefunden! Pfade prüfen.")
    sys.exit(1)

print(f"Training startet mit {len(active_clients)} Clients.")

print("\nClient Data Stats:")
print("=" * 115)
print(f"{'CLIENT ID':<20} | {'TRAIN (Skewed)':<32} | {'VAL (Local Skew)':<28} | {'TEST (Shared/Unified)':<28}")
print("=" * 115)


global_total_pneu = 0
global_total_norm = 0


def get_stats_str_for_main(loader):
    if loader is None:
        return "N/A"
    try:
        labels = get_labels_from_dataset(loader.dataset)
        n_total = len(labels)
        if n_total == 0: return "Empty"

        n_pneu = sum(labels)
        n_norm = n_total - n_pneu

        ratio = min(n_pneu, n_norm) / n_total if n_total > 0 else 0.0
        return f"{n_total} ({n_pneu}:{n_norm} | MinR={ratio:.2f})"
    except Exception:
        return f"Err"


for cid in active_clients:
    s_train = get_stats_str_for_main(client_train_loaders.get(cid))
    s_val = get_stats_str_for_main(client_val_loaders.get(cid))
    s_test = get_stats_str_for_main(client_test_loaders.get(cid))

    print(f"{cid:<20} | {s_train:<32} | {s_val:<28} | {s_test:<28}")

    train_labels = get_labels_from_dataset(client_train_loaders[cid].dataset)
    current_pneu = sum(train_labels)
    current_norm = len(train_labels) - current_pneu

    global_total_pneu += current_pneu
    global_total_norm += current_norm

print("=" * 115 + "\n")

if global_total_pneu > 0:
    pos_weight_value = global_total_norm / global_total_pneu
else:
    pos_weight_value = 1.0

print(f"[AUTO-CALC] Global Stats: Normal={global_total_norm}, Pneu={global_total_pneu}")
print(f"[AUTO-CALC] Calculated pos_weight: {pos_weight_value:.4f}")

pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)

skewed_clients = []
for cid, loader in client_train_loaders.items():
    try:
        labels = get_labels_from_dataset(loader.dataset)
        if len(labels) > 0:
            ratio = sum(labels) / len(labels)
            if ratio < 0.2 or ratio > 0.8:
                skewed_clients.append(cid)
    except:
        pass

if skewed_clients:
    print(f"[INFO] Skewed Clients detected (MinR < 0.2): {skewed_clients}")
else:
    print("[INFO] No heavily skewed clients detected.")
print("-" * 115 + "\n")


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

global_train_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_train_loaders.values()]),
    batch_size=32, shuffle=True, collate_fn=None
)
global_test_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_test_loaders.values()]),
    batch_size=64, shuffle=False, collate_fn=None
)

qfl_global_model_metrics = {}
qfl_client_metrics = {}
qfl_local_model_metrics = {}
qfl_external_metrics = {}
baseline_client_metrics = {}
baseline_external_metrics = {}
central_global_metrics = {}

all_train_accuracies = {seed: {cid: [] for cid in active_clients} for seed in seeds}
all_baseline_train_accuracies = {seed: {cid: [] for cid in active_clients} for seed in seeds}


print("\n# STARTING QUANTUM FEDERATED LEARNING (QFL) #")
for seed in seeds:
    print(f"\n--- SEED {seed} ---")
    set_seed(seed)

    global_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    client_models = {cid: QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device) for cid
                     in active_clients}

    optimizers = {cid: torch.optim.Adam(model.parameters(), lr=learning_rate) for cid, model in client_models.items()}

    print(model.parameters())

    schedulers = {
        cid: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[cid],
            mode='min',
            factor=0.5,
            patience=10,
        )
        for cid in client_models.keys()
    }

    for round_num in range(num_rounds):
        print(f"--- Round {round_num + 1}/{num_rounds} ---")
        client_updates = []
        for cid in active_clients:
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())

            train_res = train_client(
                model=local_model,
                client_train_loader=client_train_loaders[cid],
                criterion=criterion,
                optimizer=optimizers[cid],
                client_id=cid,
                num_epochs=local_epochs,
                client_val_loaders=client_val_loaders,
                print_last_only=False,
                last_round=(round_num == num_rounds - 1),
            )

            client_updates.append(train_res["weights"])
            all_train_accuracies[seed][cid].extend(train_res["train_accuracies"])

            final_train_loss = train_res["train_losses"][-1]
            schedulers[cid].step(final_train_loss)

        global_model.load_state_dict(aggregate_models(global_model, client_updates))

    qfl_client_metrics[seed] = {}
    qfl_local_model_metrics[seed] = {}
    qfl_external_metrics[seed] = {}

    for cid in active_clients:
        res_local = evaluate_model(client_models[cid], client_test_loaders[cid], criterion)
        qfl_local_model_metrics[seed][cid] = {
            "accuracy": res_local[0], "loss": res_local[1], "f1": res_local[2],
            "auc": res_local[3], "auc_pr": res_local[4], "precision": res_local[5],
            "recall": res_local[6], "confusion_matrix": res_local[7],
            "probs": res_local[8], "labels": res_local[9], **res_local[10]
        }
        print(f"[Local->{cid}] ACC={res_local[0]:.2f}% | F1={res_local[2]:.3f}")

        res_global = evaluate_model(global_model, client_test_loaders[cid], criterion)
        qfl_client_metrics[seed][cid] = {
            "accuracy": res_global[0], "loss": res_global[1], "f1": res_global[2],
            "auc": res_global[3], "auc_pr": res_global[4], "precision": res_global[5],
            "recall": res_global[6], "confusion_matrix": res_global[7],
            "probs": res_global[8], "labels": res_global[9], **res_global[10]
        }
        print(f"[Global->{cid}] ACC={res_global[0]:.2f}% | F1={res_global[2]:.3f}")

    if external_loader:
        res_ext = evaluate_model(global_model, external_loader, criterion)
        qfl_external_metrics[seed]["global_model"] = {"accuracy": res_ext[0], "f1": res_ext[2], "auc": res_ext[3]}

    res_glob = evaluate_model(global_model, global_test_loader, criterion)
    qfl_global_model_metrics[seed] = {"accuracy": res_glob[0]}


print("\n# STARTING LOCAL BASELINE TRAINING #")
for seed in seeds:
    print(f"\n--- BASELINE SEED {seed} ---")
    set_seed(seed)
    baseline_client_metrics[seed] = {}
    baseline_external_metrics[seed] = {}

    for cid in active_clients:
        local_model = QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)

        # Baseline Training (kein FedProx -> mu=0.0 gesetzt)
        train_res = train_client(
            local_model, client_train_loaders[cid], criterion, optimizer, cid,
            num_epochs=baseline_epochs, client_val_loaders=client_val_loaders,
            print_last_only=True
        )
        all_baseline_train_accuracies[seed][cid] = train_res["train_accuracies"]

        res = evaluate_model(local_model, client_test_loaders[cid], criterion)
        baseline_client_metrics[seed][cid] = {
            "accuracy": res[0], "loss": res[1], "f1": res[2],
            "auc": res[3], "auc_pr": res[4], "precision": res[5],
            "recall": res[6], "confusion_matrix": res[7],
            "probs": res[8], "labels": res[9], **res[10]
        }

        if external_loader:
            res_ext = evaluate_model(local_model, external_loader, criterion)
            baseline_external_metrics[seed][cid] = {"accuracy": res_ext[0], "f1": res_ext[2], "auc": res_ext[3]}

    central_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    central_optimizer = torch.optim.Adam(central_model.parameters(), lr=learning_rate)

    train_client(
        central_model, global_train_loader, criterion, central_optimizer, "central",
        num_epochs=baseline_epochs, client_val_loaders=None, print_last_only=True
    )

    if external_loader:
        res_c = evaluate_model(central_model, external_loader, criterion)
        central_global_metrics[seed] = {
            "accuracy": res_c[0], "loss": res_c[1], "f1": res_c[2],
            "auc": res_c[3], "auc_pr": res_c[4], "precision": res_c[5], "recall": res_c[6]
        }
    else:
        central_global_metrics[seed] = {"accuracy": 0}

print("\n# Generating Plots #")

print("\n[1/7] Training Accuracy over Epochs...")
plot_qfl_vs_baseline_accuracy(all_train_accuracies, all_baseline_train_accuracies, active_clients)

print("\n[2/7] Per-Client Boxplots...")
plot_per_client_boxplots(qfl_client_metrics, baseline_client_metrics, active_clients)

print("\n[3/7] Global Performance Distribution...")
central_acc_mean = np.nanmean([v['accuracy'] for v in central_global_metrics.values()]) if central_global_metrics else 0
plot_global_boxplot(
    qfl_client_metrics,
    baseline_client_metrics,
    active_clients,
    central_mean=central_acc_mean
)

print("\n[4/7] Model Stability...")
plot_stability_boxplot(qfl_global_model_metrics, baseline_client_metrics, central_global_metrics)

print("\n[5/7] Minority-Class F1 Analysis...")
plot_minority_f1_boxplots(qfl_client_metrics, baseline_client_metrics, active_clients, client_train_loaders)

print("\n[6/7] F1 vs. Minority Ratio...")
plot_f1_vs_minority_ratio(qfl_client_metrics, baseline_client_metrics, client_train_loaders, seeds)

print("\n[7/7] Minority F1 Density...")
plot_minority_f1_density(qfl_client_metrics, baseline_client_metrics, client_train_loaders, seeds)

if skewed_clients:
    print(f"\nAnalysing Skewed Clients: {skewed_clients}")
    plot_auc_heatmap_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)
    plot_auc_pr_heatmap_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)
    plot_confusion_matrix_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)
    plot_roc_pr_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients, seeds)

print("\n[Final] ROC/PR Overview...")
plot_roc_pr_comparison_all_clients(qfl_client_metrics, baseline_client_metrics, client_train_loaders, seeds)

print("\n[Final] Bias Summary...")
df_bias = compute_bias_summary(
    qfl_client_metrics, baseline_client_metrics, qfl_external_metrics,
    central_global_metrics, seeds, client_train_loaders
)

print("\nAnalysis Complete.")