import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from SDCCA import SparseDeepCCA
from timeit import default_timer as timer
import utils
from pathlib import Path
from utils import XNet, YNet
import zipfile
import os


def train_step(net, funcs_opt, gates_opt, x, y):
    funcs_opt.zero_grad()
    gates_opt.zero_grad()
    loss = net(x, y).mean()
    loss.backward()  # retain_graph=True)
    funcs_opt.step()
    gates_opt.step()
    return loss


def plot_gates(net, name):
    g_x, g_y = net.module.get_gates()
    g_x = g_x.cpu().detach().numpy().T
    np.save('gates.npy', g_x)
    plt.imshow(g_x.reshape(28, 28))
    plt.colorbar()
    plt.title(f'x gates,  {name}')
    plt.savefig(f'x_gates/x_gates_{name}.png')
    plt.close()
    g_y = g_y.cpu().detach().numpy()
    np.save('gates.npy', g_y)
    plt.imshow(g_y.reshape(28, 28))
    plt.title(f'y gates,  {name}')
    plt.savefig(f'y_gates/y_gates_{name}.png')
    plt.close()




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
    # labels = np.concatenate((train[:,784] ,test[:,784]),axis = 0)
    data = data.reshape(data.shape[0], 28, 28)
    data = np.expand_dims(data, axis=1)
    print('original data shape is -> ' + str(data.shape))

    return data


def main(args):
    device = torch.device(f'cuda:{args.cuda[0]}' if torch.cuda.is_available() else 'cpu')
    x, y = load_data('mnist_background_images'), load_data('mnist_background_random')

    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    x_net = XNet()
    y_net = XNet()
    net = SparseDeepCCA([x.shape[2], x.shape[3]], [y.shape[2], y.shape[3]], x_net, y_net, args.lamx, args.lamy)
    utils.print_parameters(net)

    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=args.cuda)
    else:
        net = nn.DataParallel(net)

    net = net.to(device)
    net.train()

    funcs_params = net.module.get_function_parameters()
    gates_params = net.module.get_gates_parameters()
    funcs_opt = optim.Adam(funcs_params, lr=1e-4)
    gates_opt = optim.Adam(gates_params, lr=1e-3)

    loss = []
    start = timer()
    for epoch in range(2000):
        loss.append(train_step(net, funcs_opt, gates_opt, x, y).item())
        if (epoch + 1) % 100 == 0:
            end = timer()
            print(f'epoch: {epoch + 1}    '
                  f'loss: {loss[-1]:.4f}    '
                  f'lam: {args.lamx}, {args.lamy}    '
                  f'time: {end - start:.2f}')
            start = end
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, f'{args.lamx}_{args.lamy}_{epoch + 1}_vad')
    plt.plot(loss)
    plt.savefig('loss.png')

    plt.close()
    torch.save(net, 'model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
                        help='gpu indexes, in [1,2,3] format',
                        type=str,
                        default="[1,2]")
    parser.add_argument('--lamx',
                        help='x gates reg',
                        type=float,
                        default=1)
    parser.add_argument('--lamy',
                        help='y gates reg',
                        type=float,
                        default=1)
    args = parser.parse_args()

    args.cuda = args.cuda.strip('][').split(',')
    args.cuda = [int(e) for e in args.cuda]
    main(args=args)
