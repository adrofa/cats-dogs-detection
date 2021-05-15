import torch.nn as nn


class VanillaConv(nn.Module):
    """Basic convolution layer with ReLU and optional batch normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.batch_norm = nn.BatchNorm2d(
                num_features=out_channels,
                momentum=None,
                affine=True,
                track_running_stats=True
            ) if batch_norm is True else False
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        return x


class VanillaLinear(nn.Module):
    """Basic linear layer with ReLU and optional dropout."""

    def __init__(self, in_features, out_features, dropout=0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x


