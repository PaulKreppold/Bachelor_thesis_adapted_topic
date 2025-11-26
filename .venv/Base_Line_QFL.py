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
from plots import (plot_per_client_boxplots, plot_qfl_vs_baseline_accuracy, plot_global_boxplot,
                   plot_stability_boxplot, plot_rsna_auc_stability, identify_skewed_clients, plot_confusion_matrix_skewed, plot_roc_pr_skewed, plot_auc_heatmap_skewed, plot_auc_pr_heatmap_skewed, compute_bias_summary)
# ======================================================
#                  MAIN LOOP (FINAL CLEAN VERSION)
# ======================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================================================
# CONFIGURATION
# ======================================================
seeds = [42, 1337]
num_rounds = 40
local_epochs = 3
baseline_epochs = num_rounds * local_epochs
learning_rate = 0.001

global_m_num_qubits = 10
global_m_num_layers = 6
client_m_num_layers = 6

BASE_CLIENTS_FRACTIONS = {
    "client_1": 1.0,
    "client_2": 1.0,
    "client_3": 1.0,
    "client_4": 1.0
}

# ======================================================
# LOAD DATA ONCE
# ======================================================
print("\n Loading all client data once...")

client_train_loaders, client_val_loaders, client_test_loaders, external_loader = load_client_data(
    client_qubits=client_qubits,
    custom_collate_fn=custom_collate_fn,
    client_fractions=BASE_CLIENTS_FRACTIONS
)

clients = [cid for cid in clients if "client_2" in cid or "client_3" in cid]

print(f"\n⚠️ TEST-MODUS: Gefilterte Clients: {clients}")

print("\n=== DATEN-INTEGRITÄTS-TEST (vor Training) ===")


try:
    print(f"Anzahl Train-Clients: {len(client_train_loaders)}")

    # Nimm 1–2 Beispiel-Clients zum Testen
    for cid, loader in list(client_train_loaders.items())[:2]:
        print(f"\n[Test] Lade erste Batch von {cid} ...")
        batch = next(iter(loader))
        imgs, labels = batch
        print(f"  Batchgröße: {imgs.shape}")
        print(f"  Labels-Typ: {type(labels)}, Beispiel-Labels: {labels[:10]}")

    # --- Test: identify_skewed_clients ---
    print("\n[Test] identify_skewed_clients() wird geprüft ...")
    skewed_clients = identify_skewed_clients(client_train_loaders, threshold=0.3)
    print(f"  Ergebnis: {skewed_clients if skewed_clients else 'Keine stark unbalancierten Clients erkannt'}")

    print("\n✅ Datentest bestanden – Training kann sicher starten.\n")

except Exception as e:
    print(f"\n❌ FEHLER beim Datentest: {e}")
    print("→ Bitte zuerst diesen Fehler beheben, bevor du das Training startest!\n")
    import sys
    sys.exit(1)  # Stoppt das Script, falls ein Datenfehler auftritt

criterion = nn.CrossEntropyLoss()

# Combined global loaders
global_train_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_train_loaders.values()]),
    batch_size=32, shuffle=True, collate_fn=custom_collate_fn
)
global_test_loader = DataLoader(
    ConcatDataset([ld.dataset for ld in client_test_loaders.values()]),
    batch_size=64, shuffle=False, collate_fn=custom_collate_fn
)

# ======================================================
# STORAGE DICTIONARIES
# ======================================================
qfl_global_model_metrics = {}
qfl_client_metrics = {}
qfl_external_metrics = {}
qfl_local_model_metrics = {}

baseline_client_metrics = {}
baseline_external_metrics = {}
central_global_metrics = {}

all_train_accuracies = {seed: {cid: [] for cid in clients} for seed in seeds}
all_baseline_train_accuracies = {seed: {cid: [] for cid in clients} for seed in seeds}

# ======================================================
# QFL TRAINING
# ======================================================
print("\n STARTING QUANTUM FEDERATED LEARNING (QFL)...\n")

for seed in seeds:
    print(f"\n================ SEED {seed} =================")
    set_seed(seed)

    # --- Initialize models ---
    global_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    client_models = {cid: QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device)
                     for cid in clients}
    optimizers = {cid: torch.optim.Adam(model.parameters(), lr=learning_rate)
                  for cid, model in client_models.items()}

    schedulers = {cid: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
                  for cid, opt in optimizers.items()}

    # --------------------------
    # QFL training rounds
    # --------------------------
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        # (Optional) Checke die aktuelle LR von Client 1 zur Kontrolle
        current_lr = schedulers[clients[0]].get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")

        client_updates = []

        for cid in clients:
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())

            # --- Training on client ---
            train_res = train_client(
                local_model,
                client_train_loaders[cid],
                criterion,
                optimizers[cid],
                cid,
                num_epochs=local_epochs,
                client_val_loaders=client_val_loaders,
                print_last_only=False,
                last_round=(round_num == num_rounds - 1)
            )

            client_updates.append(train_res["weights"])
            all_train_accuracies[seed][cid].extend(train_res["train_accuracies"])

            schedulers[cid].step()

        # --- Aggregate client updates into global model ---
        global_model.load_state_dict(aggregate_models(global_model, client_updates))

    # --------------------------
    # QFL evaluation
    # --------------------------
    qfl_client_metrics[seed] = {}
    qfl_external_metrics[seed] = {}
    qfl_local_model_metrics[seed] = {}

    print("\n [QFL EVALUATION RESULTS]")

    # --- Local model (each client) evaluation ---
    for cid, test_loader in client_test_loaders.items():
        (acc_local, loss_local, f1_local, auc_local, auc_pr_local,
         prec_local, rec_local, cm_local, _, _, _) = evaluate_model(local_model, test_loader, criterion)

        qfl_local_model_metrics[seed][cid] = {
            "accuracy": acc_local, "loss": loss_local, "f1": f1_local,
            "auc": auc_local, "auc_pr": auc_pr_local,
            "precision": prec_local, "recall": rec_local
        }

        print(f"  [QFL Local {cid}] ACC={acc_local:.2f}% | F1={f1_local:.3f} | AUC={auc_local:.3f}")

    # --- Global model → client testsets ---
    for cid, test_loader in client_test_loaders.items():
        (acc_global, loss_global, f1_global, auc_global, auc_pr_global,
         prec_global, rec_global, cm_global, all_probs_global,
         all_labels_global, class_f1_global) = evaluate_model(global_model, test_loader, criterion)

        qfl_client_metrics[seed][cid] = {
            "accuracy": acc_global, "loss": loss_global, "f1": f1_global,
            "auc": auc_global, "auc_pr": auc_pr_global,
            "precision": prec_global, "recall": rec_global,
            "confusion_matrix": cm_global,
            "probs": all_probs_global, "labels": all_labels_global,
            **class_f1_global
        }

        print(f"  [Global→{cid}] ACC={acc_global:.2f}% | F1={f1_global:.3f} | AUC={auc_global:.3f} | "
              f"P={prec_global:.3f} | R={rec_global:.3f}")

    # --- Global combined test set ---
    (acc_g_all, loss_g_all, f1_g_all, auc_g_all, auc_pr_g_all,
     prec_g_all, rec_g_all, cm_g_all, _, _, _) = evaluate_model(global_model, global_test_loader, criterion)

    qfl_global_model_metrics[seed] = {
        "accuracy": acc_g_all, "loss": loss_g_all, "f1": f1_g_all,
        "auc": auc_g_all, "auc_pr": auc_pr_g_all,
        "precision": prec_g_all, "recall": rec_g_all,
        "confusion_matrix": cm_g_all
    }

    print(f"\n Global model (combined test) → ACC={acc_g_all:.2f}% | F1={f1_g_all:.3f} | AUC={auc_g_all:.3f}")

    # --- External dataset evaluation ---
    (acc_e, loss_e, f1_e, auc_e, auc_pr_e,
     prec_e, rec_e, cm_e, _, _, _) = evaluate_model(global_model, external_loader, criterion)

    qfl_external_metrics[seed]["global_model"] = {
        "accuracy": acc_e, "loss": loss_e, "f1": f1_e,
        "auc": auc_e, "auc_pr": auc_pr_e,
        "precision": prec_e, "recall": rec_e,
        "confusion_matrix": cm_e
    }

    print(f"\n External dataset (global model): ACC={acc_e:.2f}% | F1={f1_e:.3f} | AUC={auc_e:.3f}")


# ======================================================
# LOCAL BASELINE TRAINING
# ======================================================
print("\n\n STARTING LOCAL BASELINE TRAINING...\n")

for seed in seeds:
    print(f"\n================ BASELINE SEED {seed} =================")
    set_seed(seed)
    baseline_client_metrics[seed] = {}
    baseline_external_metrics[seed] = {}

    for cid in clients:
        local_model = QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)

        train_res = train_client(
            local_model,
            client_train_loaders[cid],
            criterion,
            optimizer,
            cid,
            num_epochs=baseline_epochs,
            client_val_loaders=client_val_loaders,
            print_last_only=True
        )

        all_baseline_train_accuracies[seed][cid] = train_res["train_accuracies"]

        # --- Local testset evaluation ---
        (acc, loss, f1, auc, auc_pr, prec, rec, cm, all_probs_base, all_labels_base, class_f1_base) = evaluate_model(local_model, client_test_loaders[cid], criterion)
        baseline_client_metrics[seed][cid] = {
            "accuracy": acc, "loss": loss, "f1": f1,
            "auc": auc, "auc_pr": auc_pr,
            "precision": prec, "recall": rec,
            "confusion_matrix": cm,
            "probs": all_probs_base,
            "labels": all_labels_base,
            **class_f1_base
        }

        print(f"  [Baseline {cid} test] ACC={acc:.2f}% | F1={f1:.3f} | AUC={auc:.3f} | P={prec:.3f} | R={rec:.3f}")

        # --- External evaluation ---
        (acc_e, loss_e, f1_e, auc_e, auc_pr_e, prec_e, rec_e, _, _, _, _) = evaluate_model(local_model, external_loader, criterion)
        baseline_external_metrics[seed][cid] = {
            "accuracy": acc_e, "loss": loss_e, "f1": f1_e,
            "auc": auc_e, "auc_pr": auc_pr_e,
            "precision": prec_e, "recall": rec_e
        }

    # --- Centralized model training on all data ---
    central_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    central_optimizer = torch.optim.Adam(central_model.parameters(), lr=learning_rate)

    train_client(
        central_model, global_train_loader, criterion, central_optimizer,
        "central", num_epochs=baseline_epochs,
        client_val_loaders=None, print_last_only=True
    )

    (acc_c, loss_c, f1_c, auc_c, auc_pr_c, prec_c, rec_c, _, _, _, _) = evaluate_model(central_model, external_loader, criterion)

    central_global_metrics[seed] = {
        "accuracy": acc_c, "loss": loss_c, "f1": f1_c,
        "auc": auc_c, "auc_pr": auc_pr_c,
        "precision": prec_c, "recall": rec_c
    }

    print(f"  Central External → ACC={acc_c:.2f}% | F1={f1_c:.3f} | AUC={auc_c:.3f}")


# ======================================================
# FINAL SUMMARY
# ======================================================
print("\n\n FINAL RESULTS SUMMARY ")
for seed in seeds:
    avg_qfl_f1 = np.mean([v["f1"] for v in qfl_client_metrics[seed].values()])
    avg_baseline_f1 = np.mean([v["f1"] for v in baseline_client_metrics[seed].values()])
    print(f"\n[Seed {seed}] QFL vs Baseline")
    print(f"  Avg QFL F1 (local tests): {avg_qfl_f1:.3f}")
    print(f"  Avg Baseline F1 (local tests): {avg_baseline_f1:.3f}")
    print(f"  External F1 QFL (global): {qfl_external_metrics[seed]['global_model']['f1']:.3f}")
    print(f"  External F1 Central: {central_global_metrics[seed]['f1']:.3f}")

# ======================================================
# PLOTTING
# ======================================================
print("\n Generating Plots...")

plot_per_client_boxplots(
    {s: {cid: m["accuracy"] for cid, m in qfl_client_metrics[s].items()} for s in qfl_client_metrics},
    {s: {cid: m["accuracy"] for cid, m in baseline_client_metrics[s].items()} for s in baseline_client_metrics},
    clients
)

plot_global_boxplot(
    {s: {cid: m["accuracy"] for cid, m in qfl_client_metrics[s].items()} for s in qfl_client_metrics},
    {s: {cid: m["accuracy"] for cid, m in baseline_client_metrics[s].items()} for s in baseline_client_metrics},
    clients,
    np.mean([v['accuracy'] for v in central_global_metrics.values()])
)

plot_stability_boxplot(
    {s: qfl_global_model_metrics[s]["accuracy"] for s in qfl_global_model_metrics},
    {s: np.mean([v["accuracy"] for v in baseline_client_metrics[s].values()]) for s in baseline_client_metrics},
    {s: central_global_metrics[s]["accuracy"] for s in central_global_metrics}
)

plot_qfl_vs_baseline_accuracy(all_train_accuracies, all_baseline_train_accuracies, clients)

skewed_clients = identify_skewed_clients(client_train_loaders, threshold=0.3)
plot_confusion_matrix_skewed(qfl_client_metrics[seed], baseline_client_metrics[seed], skewed_clients)
plot_roc_pr_skewed(qfl_client_metrics[seed], baseline_client_metrics[seed], skewed_clients)
plot_auc_heatmap_skewed(qfl_client_metrics[seed], baseline_client_metrics[seed], skewed_clients)
plot_auc_pr_heatmap_skewed(qfl_client_metrics[seed], baseline_client_metrics[seed], skewed_clients)

df_bias = compute_bias_summary(
    qfl_client_metrics, baseline_client_metrics,
    qfl_external_metrics, central_global_metrics, seeds
)
