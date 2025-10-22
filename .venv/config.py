import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clients = ['client_1', 'client_2','client_3', 'client_4']

client_qubits = {'client_1': 10, 'client_2': 10,'client_3': 10, 'client_4': 10}
