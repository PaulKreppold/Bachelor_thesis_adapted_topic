import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_clients = ["client_1", "client_2", "client_3", "client_4"]

clients = [f"{base_client}_sub{j}" for base_client in base_clients for j in range(1, 5)]

client_qubits = {client: 10 for client in clients}
