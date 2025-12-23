import torch
import torch.optim as optim
import numpy as np
import random
import os
import copy

from config import device, clients, seeds, num_layers, learning_rate, batch_size, qfl_num_rounds, qfl_epochs, \
    baseline_epochs
from Base_Line import QuantumModel
from train_and_eval import train_client, evaluate_model
from aggregation_method import aggregate_models
from data_prep import get_all_client_loaders, print_class_distributions
from plots import run_full_evaluation_suite


def set_seed(seed):
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    client_loaders = get_all_client_loaders(batch_size=batch_size)
    print_class_distributions(client_loaders)

    qfl_res, base_res = {s: {} for s in seeds}, {s: {} for s in seeds}
    qfl_hists, base_hists = {s: {} for s in seeds}, {s: {} for s in seeds}

    for seed in seeds:
        print(f"\n{'#' * 60}\n# START SEED {seed}\n{'#' * 60}")

        # BASELINE
        for cid in clients:
            set_seed(seed)
            model = QuantumModel(num_qubits=10, num_layers=num_layers).to(device)
            opt = optim.Adam(model.parameters(), lr=learning_rate)

            res = train_client(model, client_loaders[cid]['train'], opt, cid, baseline_epochs,
                               client_loaders[cid]['val'], device, "Baseline")
            base_res[seed][cid] = evaluate_model(model, client_loaders[cid]['test'], device)
            base_hists[seed][cid] = res["history"]

        # QFL
        set_seed(seed)
        global_model = QuantumModel(num_qubits=10, num_layers=num_layers).to(device)
        for cid in clients: qfl_hists[seed][cid] = {"train": {"accuracy": [], "loss": []}}

        for r in range(qfl_num_rounds):
            updates = []
            for cid in clients:
                local_m = copy.deepcopy(global_model).to(device)
                opt = optim.Adam(local_m.parameters(), lr=learning_rate)
                res = train_client(local_m, client_loaders[cid]['train'], opt, cid, qfl_epochs,
                                   client_loaders[cid]['val'], device, f"QFL-R{r + 1}")
                updates.append(res["weights"])
                qfl_hists[seed][cid]["train"]["accuracy"].extend(res["history"]["train"]["accuracy"])
                qfl_hists[seed][cid]["train"]["loss"].extend(res["history"]["train"]["loss"])

            global_model.load_state_dict(aggregate_models(global_model, updates))

        for cid in clients:
            qfl_res[seed][cid] = evaluate_model(global_model, client_loaders[cid]['test'], device)

    run_full_evaluation_suite(qfl_res, base_res, qfl_hists, base_hists, clients, client_loaders, seeds)


if __name__ == "__main__":
    main()