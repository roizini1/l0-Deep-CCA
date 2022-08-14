# import torch
from torch import nn
from stg import StochasticGates
# from utils import XNet  # , YNet

class DCCA(nn.Module):
    def __init__(self, net_hp):
        """
        f: X_sub_net
        g: Y_sub_net
        """
        super().__init__()
        self.net_hp = net_hp
        # self.XNet_ = XNet()
        # self.YNet_ = XNet()
        self.f = StochasticGates(features_size= net_hp.x_dim[0],  # [net_hp.x_dim[0], net_hp.x_dim[1]],
                                 sigma=net_hp.sigma_x,
                                 lambda_=net_hp.lambda_x)  # self._create_network([net_hp.x_dim[0], net_hp.x_dim[1]],
        # self.XNet_, net_hp.lambda_x, net_hp.sigma_x)

        self.g = StochasticGates(features_size=net_hp.y_dim[0],  # [net_hp.y_dim[0], net_hp.y_dim[1]],
                                 sigma=net_hp.sigma_y,
                                 lambda_=net_hp.lambda_y)  # self._create_network([net_hp.y_dim[0], net_hp.y_dim[1]],
        # self.YNet_, net_hp.lambda_y, net_hp.sigma_y)

        self.C_t_1 = None
        self.N_factor = 0

    def forward(self, X, Y):
        """
        forward pass in l0-DCCA
        :param X: 1st modality N (samples) x D_x (features)
        :param Y: 2nd modality N (samples) x D_y (features)
        :return: loss function
        """
        X_hat, Y_hat = self.f(X), self.g(Y)
        return X_hat, Y_hat  # self.XNet_(X_hat), self.YNet_(Y_hat)  # -self._get_corr(X_hat, Y_hat) + self.f.get_regularization() + self.g.get_regularization()
    '''
    @staticmethod
    def _create_network(in_features, net, lam, sigma):
        """
        create the full network for each modality
        :param in_features: number of feature (and shape) of the gates
        :param net: the non-linearity to be used
        :param lam: regularize the gates
        :param sigma: the std of the gates
        :return: sequential model in which there are gates before the non-linearity
        """
        return nn.Sequential(StochasticGates(in_features, sigma, lam),
                             net)
    '''