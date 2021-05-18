import sys

import timm
import torch.nn as nn
import torch


def get_model(version, model_weights=None, verbose=True):
    """Returns model of the provided version.
    Args:
        version (str): model version.
        model_weights (str): path to *.pt file containing model.state_dict().
        verbose (bool): if True - print total trainable params into stdout.
    """

    if version == "v1":
        # Pretrained EfficientNet-B0 with unfreezed classifier only

        model = timm.create_model('efficientnet_b0', pretrained=False if model_weights else True)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier.in_features, out_features=640),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=640, out_features=320),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=320, out_features=5),
        )
        if model_weights:
            model.load_state_dict(torch.load(model_weights, map_location="cpu"))

    elif version == "v2":
        # get model v1:
        # load weights obtained on the 1st run (unfreezed only FC layers)
        # unfreeze 2 last CONV layers

        model = get_model("v1", model_weights=model_weights, verbose=False)
        # freeze all
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze needed
        children = [c for c in model.children()]
        children = (
                children[-5:] +
                [[c for c in children[-6].children()][-1]]  # last InvertedResidual layer
        )
        for c in children:
            for p in c.parameters():
                p.requires_grad = True

    else:
        raise Exception(f"Model version '{version}' is unknown!")

    if verbose:
        print(
            "Total trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            file=sys.stdout
        )

    return model
