"""Copyright(c) 2023 your_name. All Rights Reserved."""
import torch
import torch.nn as nn
from collections import OrderedDict
from ...core import register

__all__ = ['GoogleNet']


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.branch_pool(x)

        outputs = [branch1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Add other layers (Inception modules, etc.) here as needed...
        self.inception1 = Inception(64)

        # Add other necessary layers...

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.inception1(x)
        # Pass through other layers...
        return x


@register()
class GoogleNetWrapper(nn.Module):
    def __init__(self, pretrained=False):
        super(GoogleNetWrapper, self).__init__()
        self.model = GoogleNet()
        if pretrained:
            # Load pretrained weights if available
            pass

    def forward(self, x):
        return self.model(x)