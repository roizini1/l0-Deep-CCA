import torch
import torch.nn as nn


class StochasticGates(nn.Module):
    def __init__(self, features_size, sigma, lambda_, gate_init=None):
        """
        inputs
        features_size: Dx / Dy according to input shape
        lambda_: regularization parameters which control the sparsify the input variables
        sigma: gaussian variance
        gate_init: gate initialization
        """
        super().__init__()
        self.features_size = features_size
        self.sigma = sigma
        self.lambda_ = lambda_

        """ user gate initialization """
        if gate_init is None:
            # mus = 0.5 * torch.ones((int(features_size[0]), int(features_size[1])))
            mus = 0.5 * torch.ones(features_size)
        else:
            mus = torch.from_numpy(gate_init)

        self.mus = nn.Parameter(mus, requires_grad=True)
        self.eps = None

    def forward(self, x):
        """ x input is N (samples) x Dx (features) """
        z = self.Bernoulli_relaxation()
        # gated canonical vectors:
        new_x = x * z
        return new_x

    def eps_init(self):
        sig = torch.randn(self.mus.size(), device=self.mus.get_device())
        # print(sig.get_device())
        eps = torch.mul(self.sigma, sig)

        return eps

    def Bernoulli_relaxation(self):
        """ Gaussian-based relaxation for the Bernoulli random variables """
        """ zi = max(0, min(1, µi + eps_i)) -> hard sigmoid function """

        self.eps = self.eps_init()
        # print(self.mus.get_device())
        # print(self.eps.get_device())
        return torch.clamp(self.mus+self.eps, 0.0, 1.0)

    def get_regularization(self):
        """ λ * E[||z||] """
        sqrt_2 = torch.sqrt(torch.tensor(2))
        return self.lambda_ * torch.sum((1 - torch.erf(- self.mus / (sqrt_2 * self.sigma)))) / 2

    def get_gates(self):
        return self.Bernoulli_relaxation()