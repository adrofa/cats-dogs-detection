import torch


def get_optimizer(version, model_parameters, optimizer_weights=None):
    if version == "adam_v1":
        # lr = Karpathy Score
        optimizer = torch.optim.Adam(params=model_parameters, lr=3e-4)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))

    elif version == "adam_v2":
        optimizer = torch.optim.Adam(params=model_parameters, lr=3e-6)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))

    elif version == "adam_v3":
        optimizer = torch.optim.Adam(params=model_parameters, lr=0.01)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))

    elif version == "adam_v4":
        optimizer = torch.optim.Adam(params=model_parameters, lr=0.0001)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))

    elif version == "adam_v5":
        optimizer = torch.optim.Adam(params=model_parameters, lr=0.001)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))

    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")

    return optimizer
