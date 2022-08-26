import torch
import torch.nn as nn


class XNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(6, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False),
        '''
        self.linear_1 = nn.Linear(28, 256)  # , dtype=torch.double)
        self.linear_2 = nn.Linear(256, 3)  # , dtype=torch.double)
        self.tanh = nn.ReLU()  # nn.Tanh()
        '''
        self.net = nn.Sequential(
            nn.Linear(28, 256, dtype=torch.double),
            nn.Tanh(),
            nn.Linear(256, 3, dtype=torch.double),
            nn.Tanh(),
        )
        '''
    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        out = self.tanh(x)
        # out = self.net(x)
        return out
