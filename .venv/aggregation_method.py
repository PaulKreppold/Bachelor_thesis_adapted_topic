import torch
from collections import OrderedDict



def aggregate_models(global_model, client_updates):

    global_dict = global_model.state_dict()
    avg_dict = OrderedDict()

    for key in global_dict.keys():
        compatible_tensors = []
        for client in client_updates:
            if key in client:
                if client[key].shape == global_dict[key].shape:
                    compatible_tensors.append(client[key].float())
                else:
                    print(f"Skipping incompatible parameter {key}: "
                          f"global_shape={global_dict[key].shape}, client_shape={client[key].shape}")

        if compatible_tensors:
            avg_dict[key] = torch.stack(compatible_tensors, 0).mean(0)
        else:
            avg_dict[key] = global_dict[key]

    return avg_dict