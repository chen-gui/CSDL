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

def subsampling3D(x, ratio=0.8):
    [t, n1, n2] = x.shape
    d = np.reshape(x, (t, n1*n2))
    mask = np.ones_like(d, dtype=np.float32)
    list = np.random.choice(n1*n2, int(n1*n2*ratio), replace=False)
    mask[:, list] = 0
    mask_out = np.reshape(mask, (t, n1, n2))
    out = mask_out * x
    return out, mask_out

def Addnoise(x, snr):
    x = np.array(x, dtype=np.float32)
    n1, n2, n3 = np.shape(x)
    x = np.reshape(x, (n1, n2*n3))

    Nx = len(x)  # 求出信号的长度
    noise = np.random.randn(n1, n2*n3) # 用randn产生正态分布随机数
    signal_power = np.sum(x*x)/Nx # 求信号的平均能量
    noise_power = np.sum(noise*noise)/Nx # 求噪声的平均能量
    noise_variance = signal_power/(math.pow(10., (snr/10))) #计算噪声设定的方差值
    noise = math.sqrt(noise_variance/noise_power)*noise # 按照噪声能量构成相应的白噪声
    y = x+noise
    y = np.reshape(y, (n1, n2, n3))
    return y

def Add_random_erratic_noise(x, snr):
    x = np.array(x, dtype=np.float32)
    n1, n2, n3 = np.shape(x)
    x = np.reshape(x, (n1, n2*n3))

    Nx = len(x)  # 求出信号的长度
    noise = np.random.randn(n1, n2*n3) # 用randn产生正态分布随机数
    signal_power = np.sum(x*x)/Nx # 求信号的平均能量
    noise_power = np.sum(noise*noise)/Nx # 求噪声的平均能量
    noise_variance = signal_power/(math.pow(10., (snr/10))) #计算噪声设定的方差值
    noise = math.sqrt(noise_variance/noise_power)*noise # 按照噪声能量构成相应的白噪声

    erra_in = np.random.randint(0, n2*n3, size=100)
    for i in range(100):
        noise[:, erra_in[i]] = noise[:, erra_in[i]]*10

    y = x+noise
    y = np.reshape(y, (n1, n2, n3))
    return y

clean = np.load('CG_Parabolic3D_SRLpapaer.npy')
print(clean.shape)
print(clean.max(), clean.min())

noisy = Addnoise(clean, 2)
np.save('trained_model_data-CG_Parabolic3D_SRLpapaer-off_grid_inter_four_times/noisy_offgrid.npy', noisy)
stepx = 4
stepy = 4
subdata = noisy[:, ::stepx, ::stepy]
np.save('trained_model_data-CG_Parabolic3D_SRLpapaer-off_grid_inter_four_times/subdata_offgrid.npy', subdata)

print(subdata.shape)
# plt.figure()
# plt.imshow(subdata[:, :, 10], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
# plt.show()

non_missing_indices = np.argwhere(subdata[0] != 10000)
print(non_missing_indices.shape)
obs_trace = subdata[:, non_missing_indices[:, 0], non_missing_indices[:, 1]]
obs_trace = obs_trace.T
print(obs_trace.shape)
non_missing_indices = np.array(non_missing_indices, dtype=np.float32)
obs_trace = np.array(obs_trace, dtype=np.float32)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(non_missing_indices).unsqueeze(1), torch.tensor(obs_trace).unsqueeze(1))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)

new_x_indices = torch.arange(0, subdata.shape[1], 0.25, dtype=torch.float32)  # 在x轴上等间隔插值
new_y_indices = torch.arange(0, subdata.shape[2], 0.25, dtype=torch.float32)  # 在y轴上等间隔插值
print(new_x_indices, new_y_indices, new_x_indices.shape, new_y_indices.shape)
new_indices = torch.stack(torch.meshgrid(new_x_indices, new_y_indices), dim=-1).view(-1, 2)
print(new_indices.shape)

test_loader = torch.utils.data.DataLoader(dataset=MyDataset(new_indices.unsqueeze(1)), batch_size=512, shuffle=False, num_workers=0, drop_last=False)

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

    if (epoch + 1) % 250 == 0:
        torch.save(net, 'trained_model_data-CG_Parabolic3D_SRLpapaer-off_grid_inter_four_times/{}.pth'.format(int(epoch + 1)))
        out_data = torch.zeros((1, 1, 100), dtype=torch.float32)
        net.eval()
        with torch.no_grad():
            loop2 = tqdm(enumerate(test_loader), total=len(test_loader))
            for batch_idx, testindex in loop2:
                testindex = testindex.to(device)
                output = net(testindex)
                out_data = torch.cat([out_data, output.detach().cpu()], 0)
        predict = out_data[1:].squeeze().numpy()
        predict = np.reshape(predict.T, (100, 80, 80))

        np.save('trained_model_data-CG_Parabolic3D_SRLpapaer-off_grid_inter_four_times/{}.npy'.format(int(epoch + 1)), predict)

        # plt.figure()
        # plt.imshow(clean[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        #
        # plt.figure()
        # plt.imshow(predict[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        #
        # plt.figure()
        # plt.imshow(clean[:, :, 40]-predict[:, :, 40], cmap=cseis(), vmin=-0.5, vmax=0.5, aspect='auto')
        #
        # plt.show()