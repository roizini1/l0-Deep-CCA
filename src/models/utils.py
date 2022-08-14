import torch
import torch.nn as nn


class XNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(200, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=x.get_device())
        return self.net(torch.unsqueeze(x, dim=1))

'''
class YNet(nn.Module):
    def __init__(self, in_dim):
        super(YNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
'''
