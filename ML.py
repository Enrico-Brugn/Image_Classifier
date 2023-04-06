import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, kernel = (3,6,5)):
        super().__init__()
        self.dim = 0
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 3, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
            nn.Softmax()
        )

    def forward(self, x):
        h = (x.shape[0]-kernel[2])
        w = (x.shape[1]-kernel[2])/2
        self.dim = x.shape;
        return self.layers(x)
