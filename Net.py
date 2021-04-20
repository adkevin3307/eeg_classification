import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activation: str) -> None:
        super(EEGNet, self).__init__()

        _activation = None
        if activation == 'ReLU':
            _activation = nn.ReLU
        if activation == 'LeakyReLU':
            _activation = nn.LeakyReLU
        if activation == 'ELU':
            _activation = nn.ELU

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            _activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            _activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)

        return x


class DeepConvNet(nn.Module):
    def __init__(self, activation: str) -> None:
        super(DeepConvNet, self).__init__()

        _activation = None
        if activation == 'ReLU':
            _activation = nn.ReLU
        if activation == 'LeakyReLU':
            _activation = nn.LeakyReLU
        if activation == 'ELU':
            _activation = nn.ELU

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5), bias=False)
        )

        self.deep_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(25),
            _activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout()
        )

        self.deep_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50),
            _activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout()
        )

        self.deep_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100),
            _activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout()
        )

        self.deep_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200),
            _activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout()
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.deep_conv_1(x)
        x = self.deep_conv_2(x)
        x = self.deep_conv_3(x)
        x = self.deep_conv_4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classify(x)

        return x
