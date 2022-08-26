import pytorch_lightning as pl
import torch
from dcca_net_pytorch import DCCA


class Net(pl.LightningModule):
    def __init__(self, net_hp):
        super().__init__()
        self.automatic_optimization = False
        self.net_hp = net_hp
        self.save_hyperparameters(net_hp)

        # pytorch net object:
        self.net = DCCA(net_hp)

        self.C_t_1 = None
        self.N_factor = 0

    def metric(self, X_hat, Y_hat):
        corr = self.correlation_sdl(X_hat, Y_hat)
        loss = -corr
        return loss

    def correlation_sdl(self, X, Y):
        psi_x = X - X.mean(axis=0)
        psi_y = Y - Y.mean(axis=0)
        self.N_factor = self.net_hp.alpha * self.N_factor + 1

        N = psi_x.size(dim=0)
        C_mini = (torch.flatten(psi_y, start_dim=1).T @ torch.flatten(psi_x, start_dim=1)).T / (N - 1)

        if self.C_t_1 is None:
            self.C_t_1 = torch.zeros_like(C_mini)
        self.C_t_1 = (self.net_hp.alpha * self.C_t_1 + C_mini) / self.N_factor
        diag = torch.diag(self.C_t_1)
        corr_sdl = torch.sum(self.C_t_1) - torch.sum(diag)
        return corr_sdl

    def forward(self, batch):
        X, Y = batch  # C:\Users\roizi\PycharmProjects\l0-Deep-CCA\src
        return self.net(X, Y)

    def training_step(self, batch):
        # opt = self.optimizers()
        # opt.zero_grad()
        # pass throw forward:
        X_hat, Y_hat = self(batch)
        loss = self.metric(X_hat, Y_hat)
        # self.manual_backward(loss, retain_graph=True)
        # opt.step()

        # tensorboard logs:
        tensorboard_logs = {'train_loss': loss}
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.net_hp.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.net_hp.lr)
        if self.net_hp.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.net_hp.lr)
        return {'optimizer': optimizer}

    '''
    def _get_corr(self, X, Y):
        """
        computes the correlation between X,Y
        :param X: 1st variable, N (samples) x d (features)
        :param Y: 2nd variable, N (samples) x d (features)
        :return: rho(X,Y)
        """
        psi_x = X - X.mean(axis=0)
        psi_y = Y - Y.mean(axis=0)

        C_yy = self._cov(psi_y, psi_y)
        C_yx = self._cov(psi_y, psi_x)
        C_xy = self._cov(psi_x, psi_y)
        C_xx = self._cov(psi_x, psi_x)

        C_yy_inv_root = self._mat_to_the_power(C_yy + torch.eye(C_yy.shape[0], device=Y.device) * 1e-3, -0.5)
        C_xx_inv = torch.inverse(C_xx + torch.eye(C_xx.shape[0], device=X.device) * 1e-3)
        M = torch.linalg.multi_dot([C_yy_inv_root, C_yx, C_xx_inv, C_xy, C_yy_inv_root])

        return self.effective_calc(torch.trace(M) / M.shape[0])
    
    def effective_calc(self, C_mini):
        if self.C_t_1 == 0:
            self.C_t_1 = torch.zeros_like(C_mini)
        C_t = self.net_hp.alpha * self.C_t_1 + C_mini

        self.N_factor = self.net_hp.alpha * self.N_factor + 1

        C_out = C_t / self.N_factor

        self.C_t_1 = C_out
        return C_out

    @staticmethod
    def _cov(psi_x, psi_y):
        """
        estimates the covariance matrix between two centered views
        :param psi_x: 1st centered view, N (samples) x d (features)
        :param psi_y: 2nd centered view, N (samples) x d (features)
        :return: covariance matrix
        """
        N = psi_x.shape[0]
        return (torch.flatten(psi_y, start_dim=1).T @ torch.flatten(psi_x, start_dim=1)).T / (N - 1)

    @staticmethod
    def _mat_to_the_power(A, arg):
        """
        raises matrix to the arg-th power using diagonalizing, where arg is signed float.
        if arg is integer, it's better to use 'torch.linalg.matrix_power()'
        :param A: symmetric matrix (must be PSD if taking even roots)
        :param arg: the power
        :return: A^(arg)
        """
        eig_values, eig_vectors = torch.linalg.eig(A)
        eig_values = eig_values.real
        eig_vectors = eig_vectors.real
        return torch.linalg.multi_dot([eig_vectors, torch.diag((eig_values + 1e-3) ** arg), torch.inverse(eig_vectors)])
    '''