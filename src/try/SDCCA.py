import torch
import torch.nn as nn
from stg import StochasticGates


class SparseDeepCCA(nn.Module):
    def __init__(self, x_dim, y_dim, x_net, y_net, lamx, lamy, sigmax=1, sigmay=1):
        """
        c'tor to l0-DCCA class
        :param x_dim: Dx
        :param y_dim: Dy
        :param x_net: non-linear function for x modality
        :param y_net: non-linear function for y modality
        :param lamx: regularizer for x gates
        :param lamy: regularizer for y gates
        :param sigmax: std for x gates
        :param sigmay: std for y gates
        """
        super().__init__()
        self.f = self._create_network(x_dim, x_net, lamx, sigmax)
        self.g = self._create_network(y_dim, y_net, lamy, sigmay)

        self.alpha = 0.1  # parameter to enter
        self.c_norm = 0
        self.last_C_accu = None

    def forward(self, X, Y):
        """
        forward pass in l0-DCCA
        :param X: 1st modality N (samples) x D_x (features)
        :param Y: 2nd modality N (samples) x D_y (features)
        :return: loss function
        """
        X_hat, Y_hat = self.f(X), self.g(Y)
        return - self.get_sdl(X_hat, Y_hat)  # + self.f[0].get_reg() + self.g[0].get_reg()

    def get_sdl(self, X, Y):
        # input->[batch size, flatten image shape]
        C_mini = self._cov(X, Y) - self.f[0].get_reg() - self.g[0].get_reg() # mini batch correlation
        if self.last_C_accu is None:
            self.last_C_accu = torch.zeros_like(C_mini)

        C_accu = self.alpha * self.last_C_accu + C_mini
        self.c_norm = self.alpha * self.c_norm + 1
        C_appx = C_accu / self.c_norm
        self.last_C_accu = C_appx

        return torch.sum(torch.abs(C_appx)) - torch.sum(torch.diagonal(torch.abs(C_appx)))

    def get_gates(self):
        """
        use this function to retrieve the gates values for each modality
        :return: gates values
        """
        return self.f[0].get_gates(), self.g[0].get_gates()

    def get_function_parameters(self):
        """
        use this function if you wish to use a different optimizer for functions and gates
        :return: learnable parameters of f and g
        """
        params = list()
        for net in [self.f, self.g]:
            params += list(net[1].parameters())
        return params

    def get_gates_parameters(self):
        """
        use this function if you wish to use a different optimizer for functions and gates
        :return: learnable parameters of the gates
        """
        params = list()
        for net in [self.f, self.g]:
            params += list(net[0].parameters())
        return params

    @staticmethod
    def _create_network(in_features, net, lam, sigma):
        """
        create the full network for each modality
        :param in_features: number of feature (and shape) of the gates
        :param net: the non-linearity to be used
        :param lam: the regularizer of the gates
        :param sigma: the std of the gates
        :return: sequential model in which there are gaets before the non-linearity
        """
        return nn.Sequential(StochasticGates(in_features, sigma, lam), net)

    @staticmethod
    def _cov(psi_x, psi_y):
        """
        estimates the covariance matrix between two centered views
        :param psi_x: 1st centered view, N (samples) x d (features)
        :param psi_y: 2nd centered view, N (samples) x d (features)
        :return: covariance matrix
        """
        N = psi_x.shape[0]
        return (psi_y.T @ psi_x).T / (N - 1)
