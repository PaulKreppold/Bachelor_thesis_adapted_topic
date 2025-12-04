import torch
import torch.nn as nn
import numpy as np
import random
import copy
from Base_Line import QuantumModel
from config import *
from data_prep import *
from train_and_eval import *
from aggregation_method import *
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.decomposition import PCA
from plots import (plot_per_client_boxplots, plot_qfl_vs_baseline_accuracy, plot_global_boxplot, plot_minority_f1_boxplots, plot_f1_vs_minority_ratio,
    plot_minority_f1_density, plot_stability_boxplot, plot_auc_heatmap_skewed, plot_auc_pr_heatmap_skewed, plot_confusion_matrix_skewed, plot_roc_pr_skewed,
    plot_roc_pr_comparison_all_clients, compute_bias_summary, identify_skewed_clients, analyze_lds_resilience)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seeds = [42, 1337, 1024, 99, 31]
num_rounds = 20
local_epochs = 3
baseline_epochs = num_rounds * local_epochs
learning_rate = 0.001

global_m_num_qubits = 10
global_m_num_layers = 4
client_m_num_layers = 4

BASE_CLIENTS_FRACTIONS = {client: 1.0 for client in base_clients}


print("\nLoading all client data...")
client_train_loaders, client_val_loaders, client_test_loaders, external_loader = load_client_data(
    client_qubits=client_qubits,
    custom_collate_fn=custom_collate_fn,
    client_fractions=BASE_CLIENTS_FRACTIONS
)

active_clients = sorted(list(client_train_loaders.keys()))
if not active_clients:
    print("Keine Clients gefunden! Pfade prüfen.")
    sys.exit(1)
print(f"Training startet mit {len(active_clients)} Clients: {active_clients}")


print("\nClient Data Stats:")
print("=" * 90)
print(f"{'CLIENT ID':<20} | {'TRAIN':<25} | {'VAL':<25} | {'TEST':<25}")
print("=" * 90)

def get_stats(loader):
    if not loader: return "N/A"
    try:
        labels = get_labels_from_dataset(loader.dataset)
        counts = Counter(labels)
        n_norm = counts.get(0, 0)
        n_pneu = counts.get(1, 0)
        ratio = n_pneu / (n_norm + 1e-6)
        return f"{len(labels):<4} ({n_pneu}:{n_norm} | R={ratio:.2f})"
    except:
        return f"{len(loader.dataset)} (?)"

for cid in active_clients:
    print(f"{cid:<20} | {get_stats(client_train_loaders.get(cid)):<25} | {get_stats(client_val_loaders.get(cid)):<25} | {get_stats(client_test_loaders.get(cid)):<25}")
print("=" * 90 + "\n")

# Skewed clients (<20% minority)
try:
    skewed_clients = identify_skewed_clients(client_train_loaders, threshold=0.2)
    print(f"Skewed Clients: {skewed_clients}")
except Exception as e:
    print(f"Fehler bei Skew-Test: {e}")
    skewed_clients = []

criterion = nn.BCELoss()
global_train_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_train_loaders.values()]),
    batch_size=32, shuffle=True, collate_fn=custom_collate_fn
)
global_test_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_test_loaders.values()]),
    batch_size=64, shuffle=False, collate_fn=custom_collate_fn
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
    client_models = {cid: QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device) for cid in active_clients}
    optimizers = {cid: torch.optim.Adam(model.parameters(), lr=learning_rate) for cid, model in client_models.items()}
    schedulers = {cid: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95) for cid, opt in optimizers.items()}

    for round_num in range(num_rounds):
        print(f"--- Round {round_num+1}/{num_rounds} ---")
        client_updates = []
        for cid in active_clients:
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())
            train_res = train_client(
                local_model, client_train_loaders[cid], criterion, optimizers[cid], cid,
                num_epochs=local_epochs, client_val_loaders=client_val_loaders,
                print_last_only=False, last_round=(round_num==num_rounds-1)
            )
            client_updates.append(train_res["weights"])
            all_train_accuracies[seed][cid].extend(train_res["train_accuracies"])
            schedulers[cid].step()
        global_model.load_state_dict(aggregate_models(global_model, client_updates))

    qfl_client_metrics[seed] = {}
    qfl_local_model_metrics[seed] = {}
    qfl_external_metrics[seed] = {}

    for cid in active_clients:
        res_local = evaluate_model(client_models[cid], client_test_loaders[cid], criterion)
        qfl_local_model_metrics[seed][cid] = {
            "accuracy": res_local[0],
            "loss": res_local[1],
            "f1": res_local[2],
            "auc": res_local[3],
            "auc_pr": res_local[4],
            "precision": res_local[5],
            "recall": res_local[6],
            "confusion_matrix": res_local[7],
            "probs": res_local[8],
            "labels": res_local[9],
            **res_local[10]
        }

        print(f"[Local->{cid}] ACC={res_local[0]:.2f}% | F1={res_local[2]:.3f}")


        res_global = evaluate_model(global_model, client_test_loaders[cid], criterion)
        qfl_client_metrics[seed][cid] = {
            "accuracy": res_global[0],
            "loss": res_global[1],
            "f1": res_global[2],
            "auc": res_global[3],
            "auc_pr": res_global[4],
            "precision": res_global[5],
            "recall": res_global[6],
            "confusion_matrix": res_global[7],
            "probs": res_global[8],
            "labels": res_global[9],
            **res_global[10]
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
        train_res = train_client(
            local_model, client_train_loaders[cid], criterion, optimizer, cid,
            num_epochs=baseline_epochs, client_val_loaders=client_val_loaders,
            print_last_only=True
        )
        all_baseline_train_accuracies[seed][cid] = train_res["train_accuracies"]

        res = evaluate_model(local_model, client_test_loaders[cid], criterion)
        baseline_client_metrics[seed][cid] = {
            "accuracy": res[0],
            "loss": res[1],
            "f1": res[2],
            "auc": res[3],
            "auc_pr": res[4],
            "precision": res[5],
            "recall": res[6],
            "confusion_matrix": res[7],
            "probs": res[8],
            "labels": res[9],
            **res[10]
        }

        if external_loader:
            res_ext = evaluate_model(local_model, external_loader, criterion)
            baseline_external_metrics[seed][cid] = {"accuracy": res_ext[0], "f1": res_ext[2], "auc": res_ext[3]}

    central_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    central_optimizer = torch.optim.Adam(central_model.parameters(), lr=learning_rate)
    train_client(central_model, global_train_loader, criterion, central_optimizer, "central", num_epochs=baseline_epochs, client_val_loaders=None, print_last_only=True)
    if external_loader:
        res_c = evaluate_model(central_model, external_loader, criterion)
        central_global_metrics[seed] = {"accuracy": res_c[0], "loss": res_c[1], "f1": res_c[2], "auc": res_c[3], "auc_pr": res_c[4], "precision": res_c[5], "recall": res_c[6]}
    else:
        central_global_metrics[seed] = {"accuracy": 0}

print("\n# Generating Plots #")

# ============================================================================
# 1. STANDARD PERFORMANCE PLOTS
# ============================================================================
print("\n[1/7] Training Accuracy over Epochs...")
plot_qfl_vs_baseline_accuracy(all_train_accuracies, all_baseline_train_accuracies, active_clients)

print("\n[2/7] Per-Client Boxplots (grouped by base client)...")
plot_per_client_boxplots(qfl_client_metrics, baseline_client_metrics, active_clients)

print("\n[3/7] Global Performance Distribution...")
central_acc_mean = np.mean([v['accuracy'] for v in central_global_metrics.values()]) if central_global_metrics else 0
plot_global_boxplot(
    qfl_client_metrics,
    baseline_client_metrics,
    active_clients,
    central_mean=central_acc_mean
)

print("\n[4/7] Model Stability across Seeds...")
plot_stability_boxplot(
    qfl_global_model_metrics,
    baseline_client_metrics,
    central_global_metrics
)

# ============================================================================
# 2. MINORITY CLASS & LABEL DISTRIBUTION SKEW ANALYSIS
# ============================================================================
print("\n[5/7] Minority-Class F1 Analysis...")
plot_minority_f1_boxplots(qfl_client_metrics, baseline_client_metrics, active_clients, client_train_loaders)

print("\n[6/7] F1 vs. Minority Ratio...")
plot_f1_vs_minority_ratio(qfl_client_metrics, baseline_client_metrics, client_train_loaders, seeds)

print("\n[7/7] Minority F1 Distribution...")
plot_minority_f1_density(qfl_client_metrics, baseline_client_metrics, client_train_loaders, seeds)

# ============================================================================
# 3. SKEWED CLIENT SPECIFIC ANALYSIS
# ============================================================================
if skewed_clients:
    print(f"\n{'=' * 80}")
    print(f"SKEWED CLIENT ANALYSIS (threshold <30% minority)")
    print(f"Skewed Clients: {skewed_clients}")
    print(f"{'=' * 80}")

    print("\n[3.1] AUC Heatmaps...")
    plot_auc_heatmap_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)
    plot_auc_pr_heatmap_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)

    print("\n[3.2] Confusion Matrices...")
    plot_confusion_matrix_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients)

    print("\n[3.3] ROC and Precision-Recall Curves...")
    plot_roc_pr_skewed(qfl_client_metrics, baseline_client_metrics, skewed_clients, seeds)

# ============================================================================
# 4. COMPREHENSIVE ROC/PR COMPARISON (ALL CLIENTS)
# ============================================================================
print("\n[4] ROC/PR Curves - All Clients Overview...")
plot_roc_pr_comparison_all_clients(qfl_client_metrics, baseline_client_metrics,
                                   client_train_loaders, seeds)

# ============================================================================
# 5. BIAS & FAIRNESS SUMMARY
# ============================================================================
print("\n[5] Computing Bias/Fairness Summary...")
df_bias = compute_bias_summary(
    qfl_client_metrics,
    baseline_client_metrics,
    qfl_external_metrics,
    central_global_metrics,
    seeds,
    client_train_loaders=client_train_loaders
)

# ============================================================================
# 6. COMPREHENSIVE LABEL DISTRIBUTION SKEW ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🔬 COMPREHENSIVE LABEL DISTRIBUTION SKEW ANALYSIS")
print("=" * 80)

lds_results = analyze_lds_resilience(
    qfl_client_metrics,
    baseline_client_metrics,
    client_train_loaders,
    seeds,
    skew_threshold=0.3
)