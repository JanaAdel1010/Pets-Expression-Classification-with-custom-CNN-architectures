import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch_pool(x)
        ], 1)

class CustomInception(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.incep1 = InceptionModule(3)
        self.incep2 = InceptionModule(88) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(88, num_classes)

    def forward(self, x):
        x = self.incep1(x)
        x = self.incep2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)