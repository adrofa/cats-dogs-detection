
import torch


def get_scheduler(version, optimizer):
    if version == "rop_v1":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.1, patience=3, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )

    elif version == "rop_v2":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.5, patience=5, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )

    elif version == "rop_v3":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.5, patience=3, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )

    elif version == "ccl_v1":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 0.00001, 0.01,
            step_size_up=5, step_size_down=None, mode='triangular',
            gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False,
            base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=True
        )

    else:
        raise Exception(f"Scheduler version '{version}' is unknown!")

    return scheduler
