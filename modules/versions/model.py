import sys

import timm
import torch.nn as nn


def get_model(version, model_weights=None, verbose=True):
    """Returns model of the provided version.
    Args:
        version (str): model version.
        model_weights (str): path to *.pt file containing model.state_dict().
        verbose (bool): if True - print total trainable params into stdout.
    """

    if version == "v1":
        # Pretrained Xception with fixed Conv layers

        if model_weights:
            raise Exception(f"Model version {version} doesn't support weights preloading!")

        model = timm.create_model('xception', pretrained=True)
        # for p in model.parameters():
        #     p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=1024),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=5)
        )

    else:
        raise Exception(f"Model version '{version}' is unknown!")

    if verbose:
        print(
            "Total trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            file=sys.stdout
        )

    return model
