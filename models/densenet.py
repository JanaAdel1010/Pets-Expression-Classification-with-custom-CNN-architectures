import torch.nn as nn
import torch

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.block1 = DenseBlock(32, 16)
        self.block2 = DenseBlock(48, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)