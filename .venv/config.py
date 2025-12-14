import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


clients = ["client_1", "client_2"]

client_qubits = {client: 10 for client in clients}

print(clients)









