import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, kernel = (3,6,5)):
        super().__init__()
        self.dim = 0
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(6560, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6)
        )

    def forward(self, x):
        return self.layers(x)
