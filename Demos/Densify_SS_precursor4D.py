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
        self.kan1 = KAN([in_channels, out_channels], base_activation=nn.Identity, grid_range=[-1, 1])

        self.kan2 = KAN([out_channels, out_channels], base_activation=nn.Identity, grid_range=[-1, 1])
    def forward(self, x):
        x1 = self.kan1(x)
        x2 = self.kan2(x1)
        return x2

class SeismoNet(nn.Module):
    def __init__(self, inchan=3, outchan=201, c1=8):
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

subdata = np.load('../Data/ss4d_field-data.npy')
data = np.array(subdata, dtype=np.float32)
print(data.shape)

factor = 4
new_x = data.shape[1] * factor
new_y = data.shape[2] * factor
new_z = data.shape[2] * factor

new_x_indices = torch.arange(0, data.shape[1], 1/factor, dtype=torch.float32)  # 在x轴上等间隔插值
new_y_indices = torch.arange(0, data.shape[2], 1/factor, dtype=torch.float32)  # 在y轴上等间隔插值
new_z_indices = torch.arange(0, data.shape[3], 1/factor, dtype=torch.float32)  # 在y轴上等间隔插值
print(new_x_indices, new_x_indices.shape, new_y_indices.shape, new_z_indices.shape)
new_indices = torch.stack(torch.meshgrid(new_x_indices, new_y_indices, new_z_indices), dim=-1).view(-1, 3)
print(new_indices.shape)

net = torch.load('../model/Densify_SS4D.pth')
net = net.to(device)

test_loader = torch.utils.data.DataLoader(dataset=MyDataset(new_indices.unsqueeze(1)), batch_size=1024, shuffle=False, num_workers=0, drop_last=False)
out_data = torch.zeros((1, 1, 201), dtype=torch.float32)
with torch.no_grad():
    net.eval()
    loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, testindex in loop2:
        testindex = testindex.to(device)
        output = net(testindex)
        out_data = torch.cat([out_data, output.detach().cpu()], 0)
predict = out_data[1:].squeeze().numpy()
print(predict.shape)

predict = np.reshape(predict.T, (201, data.shape[1]*factor, data.shape[2]*factor, data.shape[3]*factor))
print(predict.shape)
# np.save('1500.npy', predict)
