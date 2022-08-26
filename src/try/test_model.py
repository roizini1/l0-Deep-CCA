import torch
import argparse
# from main import load_data
import numpy as np
from sklearn.cluster import KMeans
import os
import zipfile
from pathlib import Path

def load_data(name='mnist_background_images'):
    zip_folder = os.path.join(Path(os.path.abspath('.')).parent.parent, 'data')

    # loading all the data from the zip folders:
    zip_name = os.path.join(zip_folder, name + '.zip')
    with zipfile.ZipFile(zip_name) as z:
        with z.open(name + '_train.amat') as f:
            train = np.loadtxt(f)
        with z.open(name + '_test.amat') as f:
            test = np.loadtxt(f)
    data = np.concatenate((train[:, :28 * 28], test[:, :28 * 28]), axis=0)
    labels = np.concatenate((train[:, 784], test[:, 784]), axis=0)

    data = data.reshape([data.shape[0], 28, 28])
    data = np.expand_dims(data, axis=1)
    print('original data shape is -> ' + str(data.shape))

    return data, labels


def main(args):
    device = torch.device(f'cuda:{args.cuda[0]}' if torch.cuda.is_available() else 'cpu')
    x, x_labels = load_data('mnist_background_images')  # , load_data('mnist_background_random')

    x = torch.Tensor(x).to(device)
    model = torch.load('model.pth')
    model.eval()
    '''
    for name, module in model.named_modules():
        print(name)
    '''
    f_module = model.get_submodule('module.f')

    out = f_module(x)
    out = out.cpu()
    kmeans_obj_1 = KMeans(n_clusters=10)
    labels_test_1 = kmeans_obj_1.fit_predict(out.detach().numpy())
    acc = sum(labels_test_1 == x_labels)/x_labels.shape[0]
    print('On data after net: ' + str(acc*100) + '% success')

    # on original data:
    x = x.cpu()
    kmeans_obj_2 = KMeans(n_clusters=10)
    labels_test_2 = kmeans_obj_2.fit_predict(x.detach().numpy().reshape(x_labels.shape[0], 28 * 28))
    acc_original = sum(labels_test_2 == x_labels)/x_labels.shape[0]
    print('On original data: ' + str(acc_original*100) + '% success')

    y, y_labels = load_data('mnist_background_random')  # load_data('mnist_background_images')  # , load_data('mnist_background_random')

    y = torch.Tensor(y).to(device)
    '''
    for name, module in model.named_modules():
        print(name)
    '''
    g_module = model.get_submodule('module.g')

    out = g_module(y)
    out = out.cpu()
    kmeans_obj_3 = KMeans(n_clusters=10)
    labels_test_3 = kmeans_obj_3.fit_predict(out.detach().numpy())
    acc = sum(labels_test_3 == y_labels) / y_labels.shape[0]
    print('On data after net: ' + str(acc * 100) + '% success')

    # on original data:
    y = y.cpu()
    kmeans_obj_4 = KMeans(n_clusters=10)
    labels_test_4 = kmeans_obj_4.fit_predict(y.detach().numpy().reshape(y_labels.shape[0], 28 * 28))
    acc_original = sum(labels_test_4 == y_labels) / y_labels.shape[0]
    print('On original data: ' + str(acc_original * 100) + '% success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
                        help='gpu indexes, in [0,1,2,3] format',
                        type=str,
                        default="[0,1,2,3]")
    parser.add_argument('--lamx',
                        help='x gates reg',
                        type=float,
                        default=1.0)
    parser.add_argument('--lamy',
                        help='y gates reg',
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    args.cuda = args.cuda.strip('][').split(',')
    args.cuda = [int(e) for e in args.cuda]
    main(args=args)
