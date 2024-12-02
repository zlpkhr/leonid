import torch
import torch.nn as nn
import torch.nn.functional as F


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2)
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.s4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.f5(x)
        x = F.relu(x)
        x = self.f6(x)
        x = F.relu(x)
        x = self.f7(x)
        return x
