import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
import numpy as np # linear algebra
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T
import math
from torchvision import transforms
import torch
import torch.nn as nn
import random
import scipy.io as sio
import matplotlib.colors as mcolors
import h5py
from scipy.io import whosmat, loadmat
from efficient_kan import KAN
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

def cseis():
    seis = np.concatenate(
        (np.concatenate((0.5 * np.ones([1, 40]), np.expand_dims(np.linspace(0.5, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose(),
            np.concatenate((0.25 * np.ones([1, 40]), np.expand_dims(np.linspace(0.25, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose(),
            np.concatenate((np.zeros([1, 40]), np.expand_dims(np.linspace(0, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose()), axis=1)
    return ListedColormap(seis)

class Doublel_kan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kan1 = KAN([in_channels, out_channels], base_activation=nn.Identity, grid_range=[-2, 2])
        self.kan2 = KAN([out_channels, out_channels], base_activation=nn.Identity, grid_range=[-2, 2])
    def forward(self, x):
        x1 = self.kan1(x)
        x2 = self.kan2(x1)

        return x2

class SeismoNet(nn.Module):
    def __init__(self, inchan=2, outchan=100, c1=4):
        super().__init__()

        self.c1 = Doublel_kan(inchan, c1)
        self.c2 = Doublel_kan(c1, c1*2)
        self.c3 = Doublel_kan(c1*2, c1*4)
        self.c4 = Doublel_kan(c1*4, c1*8)
        self.c5 = Doublel_kan(c1*8, c1*16)
        self.c6 = Doublel_kan(c1*16, outchan)

    def forward(self, t):
        t0 = self.c1(t)
        t0 = self.c2(t0)
        t0 = self.c3(t0)
        t0 = self.c4(t0)
        t0 = self.c5(t0)
        t0 = self.c6(t0)
        return t0

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X):
        super().__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

clean = np.load('syn3d.npy')
print(clean.shape)
print(clean.max(), clean.min())
subdata = np.load('syn3d-subdata.npy')
mask = np.load('syn3d-mask.npy')

lossdata = np.copy(subdata)

non_missing_indices = np.argwhere(mask[0] == 1)
print(non_missing_indices.shape)
obs_trace = subdata[:, non_missing_indices[:, 0], non_missing_indices[:, 1]]
obs_trace = obs_trace.T
print(obs_trace.shape)

non_missing_indices = np.array(non_missing_indices, dtype=np.float32)
obs_trace = np.array(obs_trace, dtype=np.float32)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(non_missing_indices).unsqueeze(1), torch.tensor(obs_trace).unsqueeze(1))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

missing_indices = np.argwhere(mask[0] != 2)
print(missing_indices.shape)
missing_indices_for_output = missing_indices
missing_indices = np.array(missing_indices, dtype=np.float32)
test_loader = torch.utils.data.DataLoader(dataset=MyDataset(torch.tensor(missing_indices).unsqueeze(1)), batch_size=512, shuffle=False, num_workers=0, drop_last=False)


net = SeismoNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=5e-5)
loss_function = nn.MSELoss(reduction='mean')
net.train()

num_epoch = 1000

for epoch in range(num_epoch):
    loop1 = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (tranindex, tranlabel) in loop1:
        tranindex = tranindex.to(device)
        tranlabel = tranlabel.to(device)
        output = net(tranindex)
        train_loss = loss_function(output, tranlabel)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        loop1.set_description(f'Train-Epoch [{epoch}/{num_epoch}]')
        loop1.set_postfix(loss=train_loss.item())

    if (epoch + 1) % 1000 == 0:
        #torch.save(net, '{}.pth'.format(int(epoch + 1)))
        out_data = torch.zeros((1, 1, 100), dtype=torch.float32)
        with torch.no_grad():
            net.eval()
            loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, testindex in loop2:
                testindex = testindex.to(device)
                output = net(testindex)
                out_data = torch.cat([out_data, output.detach().cpu()], 0)
        predict = out_data[1:].squeeze().numpy()

        subdata[:, missing_indices_for_output[:, 0], missing_indices_for_output[:, 1]] = predict.T

        #np.save('{}.npy'.format(int(epoch + 1)), subdata)

        plt.figure()
        plt.imshow(lossdata[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        plt.figure()
        plt.imshow(subdata[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        plt.figure()
        plt.imshow(clean[:, :, 40]-subdata[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        plt.show()
