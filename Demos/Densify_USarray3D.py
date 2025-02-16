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

class Doublel_kan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kan1 = KAN([in_channels, out_channels], base_activation=nn.Identity, grid_range=[-0.2, 0.2])

        self.kan2 = KAN([out_channels, out_channels], base_activation=nn.Identity, grid_range=[-0.2, 0.2])
    def forward(self, x):
        x1 = self.kan1(x)
        x2 = self.kan2(x1)

        return x2

class KAN5D(nn.Module):
    def __init__(self, inchan=2, outchan=5400, c1=8):
        super().__init__()

        self.c1 = Doublel_kan(inchan, c1)
        self.c2 = Doublel_kan(c1, c1*4)
        self.c3 = Doublel_kan(c1*4, c1*16)
        self.c4 = Doublel_kan(c1*16, c1*32)
        self.c5 = Doublel_kan(c1*32, c1*64)
        self.c6 = Doublel_kan(c1*64, c1*128)
        self.c7 = Doublel_kan(c1*128, c1*256)
        self.c8 = Doublel_kan(c1*256, outchan)

    def forward(self, t):
        t0 = self.c1(t)
        t0 = self.c2(t0)
        t0 = self.c3(t0)
        t0 = self.c4(t0)
        t0 = self.c5(t0)
        t0 = self.c6(t0)
        t0 = self.c7(t0)
        t0 = self.c8(t0)

        return t0

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, X):
        super().__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

subdata = np.load('USarray3D-data.npy')
print(subdata.shape)
factor = 4
new_x = subdata.shape[1] * factor
new_y = subdata.shape[2] * factor

new_x_indices = torch.arange(0, subdata.shape[1], 1/factor, dtype=torch.float32)  # 在x轴上等间隔插值
new_y_indices = torch.arange(0, subdata.shape[2], 1/factor, dtype=torch.float32)  # 在y轴上等间隔插值
print(new_x_indices, new_x_indices.shape, new_y_indices.shape)

new_indices = torch.stack(torch.meshgrid(new_x_indices, new_y_indices), dim=-1).view(-1, 2)
print(new_indices.shape)

net = torch.load('2000.pth')
net = net.to(device)

test_loader = torch.utils.data.DataLoader(dataset=MyDataset(new_indices.unsqueeze(1)), batch_size=32, shuffle=False, num_workers=0, drop_last=False)
out_data = torch.zeros((1, 1, 5400), dtype=torch.float32)
with torch.no_grad():
    net.eval()
    loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, testindex in loop2:
        testindex = testindex.to(device)
        output = net(testindex)
        out_data = torch.cat([out_data, output.detach().cpu()], 0)
predict = out_data[1:].squeeze().numpy()
print(predict.shape)

predict = np.reshape(predict.T, (5400, 16 * factor, 28 * factor))
print(predict.shape)

# np.save('pro-2000_desified_factor_4.npy', predict)

