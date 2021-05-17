import torch


def get_optimizer(version, model_parameters, optimizer_weights=None):
    if version == "adam_v1":
        optimizer = torch.optim.Adam(params=model_parameters, lr=3e-4)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))
        # lr = Karpathy Score

    elif version == "adam_v2":
        optimizer = torch.optim.Adam(params=model_parameters, lr=3e-5)
        if optimizer_weights:
            optimizer.load_state_dict(torch.load(optimizer_weights))
        # lr = Karpathy Score

    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")

    return optimizer
