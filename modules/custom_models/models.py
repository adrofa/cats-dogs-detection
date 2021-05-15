from modules.custom_models.layers import VanillaConv, VanillaLinear

import torch
import torch.nn as nn


class CustomModelV1(nn.Module):
    def __init__(self, batch_norm=False, dropout=False):
        super().__init__()
        self.img_size = 256  # height/width of an input image

        self.conv1 = VanillaConv(in_channels=3, out_channels=64,
                                 kernel_size=10, stride=5, padding=0,
                                 batch_norm=batch_norm)
        self.conv2 = VanillaConv(in_channels=64, out_channels=32,
                                 kernel_size=6, stride=3, padding=0,
                                 batch_norm=batch_norm)
        self.conv3 = VanillaConv(in_channels=32, out_channels=16,
                                 kernel_size=3, stride=1, padding=0,
                                 batch_norm=batch_norm)

        self.fc1 = VanillaLinear(in_features=2704, out_features=676, dropout=dropout)
        self.fc2 = VanillaLinear(in_features=676, out_features=338, dropout=dropout)

        self.head = nn.Linear(in_features=338, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        x = self.head(x)
        return x
