import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_clients = ["client_1", "client_2", "client_3"]
num_subs_per_client = 4

clients = [f"{main}_sub_{i}" for main in main_clients for i in range(num_subs_per_client)]
client_qubits = {client: 10 for client in clients}

sub_to_main_mapping = {
    f"{main}_sub_{i}": main for main in main_clients for i in range(num_subs_per_client)
}

qfl_num_rounds = 12
qfl_epochs = 4

seeds = [42, 1337, 2024]
baseline_epochs = qfl_num_rounds * qfl_epochs

learning_rate = 0.01
batch_size = 32
num_layers = 6





