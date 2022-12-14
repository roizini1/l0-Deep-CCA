# import torch
from torch import nn
from stg import StochasticGates
from utils import XNet


class DCCA(nn.Module):
    def __init__(self, net_hp):
        """
        f: X_sub_net
        g: Y_sub_net
        """
        super().__init__()
        self.net_hp = net_hp

        # net layers
        self.XNet_ = XNet()
        self.YNet_ = XNet()
        self.f = StochasticGates(features_size=list(net_hp.x_dim),  # [net_hp.x_dim[0], net_hp.x_dim[1]],
                                 sigma=net_hp.sigma_x,
                                 lambda_=net_hp.lambda_x)  # self._create_network([net_hp.x_dim[0], net_hp.x_dim[1]],

        self.g = StochasticGates(features_size=list(net_hp.y_dim),  # [net_hp.y_dim[0], net_hp.y_dim[1]],
                                 sigma=net_hp.sigma_y,
                                 lambda_=net_hp.lambda_y)  # self._create_network([net_hp.y_dim[0], net_hp.y_dim[1]],

    def forward(self, X, Y):
        """
        forward pass in l0-DCCA
        :param X: 1st modality N (samples) x Dx (features)
        :param Y: 2nd modality N (samples) x Dy (features)
        :return: sdl loss function
        """
        X_hat, Y_hat = self.f(X).float(), self.g(Y).float()
        x_out, y_out = self.XNet_(X_hat), self.YNet_(Y_hat)

        return x_out, y_out
