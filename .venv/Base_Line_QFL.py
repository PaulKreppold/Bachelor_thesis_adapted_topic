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
from plots import plot_per_client_boxplots, plot_qfl_vs_baseline_accuracy, plot_global_boxplot, plot_stability_boxplot


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seeds = [42, 1337, 2024, 99, 123]
num_rounds = 10
local_epochs = 10
baseline_epochs = num_rounds * local_epochs
learning_rate = 0.01

global_m_num_qubits = 10
global_m_num_layers = 6
client_m_num_layers = 6

central_baseline_results_across_all_seeds = {}
results_across_all_seeds = {}
baseline_results_across_all_seeds = {}
qfl_results_global_model_on_clients_testset = {}
baseline_client_results = {}
all_train_accuracies = {seed: {client_id: [] for client_id in clients} for seed in seeds}
all_baseline_train_accuracies = {seed: {client_id: [] for client_id in clients} for seed in seeds}

print("STARTING QUANTUM FEDERATED LEARNING (QFL) PROCESS...")

for seed in seeds:
    print(f"\n================ SEED {seed} =================")
    set_seed(seed)

    global_model = QuantumModel(num_qubits=global_m_num_qubits,
                                num_layers=global_m_num_layers).to(device)
    client_models = {cid: QuantumModel(num_qubits=client_qubits[cid],
                                       num_layers=client_m_num_layers).to(device)
                     for cid in clients}
    optimizers = {cid: torch.optim.Adam(model.parameters(), lr=learning_rate)
                  for cid, model in client_models.items()}

    fractions = {"client_1": 0.13, "client_2": 0.13, "client_3": 0.1, "client_4": 0.13}
    client_train_loaders, client_val_loaders, client_test_loaders = load_client_data(client_qubits, fraction=fractions)
    criterion = nn.BCEWithLogitsLoss()

    all_test_datasets = [loader.dataset for loader in client_test_loaders.values()]
    global_test_dataset = ConcatDataset(all_test_datasets)
    global_test_loader = DataLoader(global_test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

    all_train_datasets = [loader.dataset for loader in client_train_loaders.values()]
    global_train_dataset = ConcatDataset(all_train_datasets)
    global_train_loader = DataLoader(global_train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num+1}/{num_rounds} ---")
        client_updates = []

        for cid in clients:
            local_model = client_models[cid]
            local_model.load_state_dict(global_model.state_dict())
            training_results = train_client(
                model=local_model,
                client_train_loader=client_train_loaders[cid],
                criterion=criterion,
                optimizer=optimizers[cid],
                client_id=cid,
                num_epochs=local_epochs,
                client_val_loaders=client_val_loaders
            )
            client_updates.append(training_results["weights"])
            all_train_accuracies[seed][cid].extend(training_results["accuracies"])

        new_global_weights = aggregate_models(global_model, client_updates)
        global_model.load_state_dict(new_global_weights)

    qfl_results_global_model_on_clients_testset[seed] = {}
    for cid, test_loader in client_test_loaders.items():
        acc, _ = evaluate_model(global_model, test_loader, criterion)
        qfl_results_global_model_on_clients_testset[seed][cid] = acc
        print(f"Global model on client {cid}: test accuracy = {acc:.2f}%")

    final_acc, _ = evaluate_model(global_model, global_test_loader, criterion)
    results_across_all_seeds[seed] = final_acc
    print(f"Global combined test accuracy = {final_acc:.2f}%")

print("\n\nSTARTING BASELINE TRAINING (LOCAL ONLY)...")
for seed in seeds:
    print(f"\n================ BASELINE SEED {seed} =================")
    set_seed(seed)
    baseline_client_results[seed] = {}
    for cid in clients:
        local_model = QuantumModel(num_qubits=client_qubits[cid], num_layers=client_m_num_layers).to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)

        training_results = train_client(
            model=local_model,
            client_train_loader=client_train_loaders[cid],
            criterion=criterion,
            optimizer=optimizer,
            client_id=cid,
            num_epochs=baseline_epochs,
            client_val_loaders=client_val_loaders
        )
        all_baseline_train_accuracies[seed][cid] = training_results["accuracies"]

        acc, _ = evaluate_model(local_model, client_test_loaders[cid], criterion)
        baseline_client_results[seed][cid] = acc
        print(f"Client {cid} Baseline test accuracy = {acc:.2f}%")

    baseline_results_across_all_seeds[seed] = np.mean(list(baseline_client_results[seed].values()))
    print(f"Average client accuracy (Seed {seed}) = {baseline_results_across_all_seeds[seed]:.2f}%")

    print(f"--- CENTRAL BASELINE CALCULATION (Seed {seed}) ---")

    central_model = QuantumModel(num_qubits=global_m_num_qubits, num_layers=global_m_num_layers).to(device)
    central_optimizer = torch.optim.Adam(central_model.parameters(), lr=learning_rate)

    central_training_results = train_client(
        model=central_model,
        client_train_loader=global_train_loader,  # Verwendet den aggregierten Loader
        criterion=criterion,
        optimizer=central_optimizer,
        client_id="Central Baseline",
        num_epochs=baseline_epochs,  # Gleiche Gesamt-Epochen wie die lokale Baseline
        client_val_loaders=None
    )

    central_acc, _ = evaluate_model(central_model, global_test_loader, criterion)

    central_baseline_results_across_all_seeds[seed] = central_acc
    print(f"Central Baseline global test accuracy (Seed {seed}) = {central_acc:.2f}%")

print("\n--- FINAL RESULTS ---")

qfl_acc_list = list(results_across_all_seeds.values())
print(f"QFL global accuracies per seed: {[f'{a:.2f}' for a in qfl_acc_list]}")
print(f"Mean ± Std: {np.mean(qfl_acc_list):.2f}% ± {np.std(qfl_acc_list):.2f}%")

baseline_acc_list = list(baseline_results_across_all_seeds.values())
print(f"Baseline avg accuracies per seed: {[f'{a:.2f}' for a in baseline_acc_list]}")
print(f"Mean ± Std: {np.mean(baseline_acc_list):.2f}% ± {np.std(baseline_acc_list):.2f}%")

central_acc_list = list(central_baseline_results_across_all_seeds.values())
print(f"Central Baseline avg accuracies per seed: {[f'{a:.2f}' for a in central_acc_list]}")
print(f"Central Baseline Mean ± Std: {np.mean(central_acc_list):.2f}% ± {np.std(central_acc_list):.2f}%")

plot_per_client_boxplots(qfl_results_global_model_on_clients_testset, baseline_client_results, clients)

central_overall_mean = np.mean(central_acc_list)
plot_global_boxplot(qfl_results_global_model_on_clients_testset, baseline_client_results, clients, central_overall_mean)

plot_stability_boxplot(results_across_all_seeds, baseline_results_across_all_seeds, central_baseline_results_across_all_seeds)

plot_qfl_vs_baseline_accuracy(all_train_accuraciesk, all_baseline_train_accuracies, clients)