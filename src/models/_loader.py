import os

import torch
import wget
import numpy as np
import zipfile
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/dsi/ziniroi/project_usl/data"):
        super().__init__()
        self.data_dir = data_dir
        self.mnist_train = None
        self.mnist_test = None
        # self.prepare_data()

    def prepare_data(self):
        path_1 = os.path.join(self.data_dir, "mnist_background_images.zip")
        if not os.path.exists(path_1):
            URL = "https://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip"
            wget.download(URL, "mnist_background_images.zip")
        else:
            print("mnist_background_images.zip is saved")
        path_2 = os.path.join(self.data_dir, "mnist_background_random.zip")
        if not os.path.exists(path_2):
            URL = "https://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip"
            wget.download(URL, "mnist_background_random.zip")
        else:
            print("mnist_background_random.zip is saved")

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            path_1 = os.path.join(self.data_dir, "mnist_background_images.zip")
            path_2 = os.path.join(self.data_dir, "mnist_background_random.zip")
            self.mnist_train = CustomDataset(path_1, path_2)
        '''
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = CustomDataset(path_1, path_2)
        '''

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=40)
    '''
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
    '''


class CustomDataset(Dataset):
    def __init__(self, data_dir_1, data_dir_2):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.len = None
        self.data = self.loader()

    def get_len(self):
        return self.len

    def loader(self):
        # loading all the data from the zip folders:
        with zipfile.ZipFile(self.data_dir_1) as z:
            with z.open('mnist_background_images_train.amat') as f:
                train = np.loadtxt(f)
            with z.open('mnist_background_images_test.amat') as f:
                test = np.loadtxt(f)
        X = np.concatenate((train[:, 0:784], test[:, 0:784]), axis=0)
        # loading all the data from the zip folders:
        with zipfile.ZipFile(self.data_dir_2) as z:
            with z.open('mnist_background_random_train.amat') as f:
                train = np.loadtxt(f)
            with z.open('mnist_background_random_test.amat') as f:
                test = np.loadtxt(f)

        Y = np.concatenate((train[:, 0:784], test[:, 0:784]), axis=0)
        # labels = np.concatenate((train[:, 784], test[:, 784]), axis=0)
        '''
        print('original data shape is -> ' + str(X.shape))
        print('original data labels shape is -> ' + str(labels.shape))
        '''
        self.len = min(X.shape[0], Y.shape[0])
        return X[:self.len, :], Y[:self.len, :]  # torch.from_numpy(X[:self.len, :].reshape([self.len, 28, 28])),
        # torch.from_numpy(Y[:self.len, :].reshape([self.len, 28, 28]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X, Y = self.data
        return X[idx, :].reshape([28, 28]), Y[idx, :].reshape([28, 28])  # X[idx, :, :], Y[idx, :, :]
