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

class KAN5D(nn.Module):
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

Seismics = {
    'red': [(0.0, 0.6666666666666666, 0.6666666666666666),
            (0.070352, 1.0, 1.0),
            (0.25, 1.0, 1.0),
            (0.5, 0.9529411764705882, 0.9529411764705882),
            (0.883249, 0.2196078431372549, 0.2196078431372549),
            (1.0, 0.0, 0.0)],
    'green': [(0.0, 0.0, 0.0),
              (0.070352, 0.10980392156862745, 0.10980392156862745),
              (0.25, 0.7843137254901961, 0.7843137254901961),
              (0.5, 0.9529411764705882, 0.9529411764705882),
              (0.883249, 0.27450980392156865, 0.27450980392156865),
              (1.0, 0.0, 0.0)],
    'blue': [(0.0, 0.0, 0.0),
             (0.070352, 0.0, 0.0),
             (0.25, 0.0, 0.0),
             (0.5, 0.9529411764705882, 0.9529411764705882),
             (0.883249, 0.4980392156862745, 0.4980392156862745),
             (1.0, 0.0, 0.0)]
}
# 使用自定义的颜色字典创建色标
Seismics_map = matplotlib.colors.LinearSegmentedColormap('Seismics', Seismics)

subdata = np.load('ss4d_field-data.npy')
data = np.array(subdata, dtype=np.float32)
print(data.shape)

raw = loadmat('./4dssmtzimaging-main/data/ss4d_field.mat')
ss4d_nmo = raw['ss4d_nmo']
ss4d_nmo = np.array(ss4d_nmo, dtype=np.float32)
print(ss4d_nmo.shape, ss4d_nmo.max(), ss4d_nmo.min())

factor = 4
# 创建新的网格
new_x = data.shape[1] * factor
new_y = data.shape[2] * factor
new_z = data.shape[2] * factor

# 创建新的索引网格
# 通过 np.meshgrid 创建新的索引（这里你可以替换为你的矩阵索引）
new_x_indices = torch.arange(0, data.shape[1], 1/factor, dtype=torch.float32)  # 在x轴上等间隔插值
new_y_indices = torch.arange(0, data.shape[2], 1/factor, dtype=torch.float32)  # 在y轴上等间隔插值
new_z_indices = torch.arange(0, data.shape[3], 1/factor, dtype=torch.float32)  # 在y轴上等间隔插值
print(new_x_indices, new_x_indices.shape, new_y_indices.shape, new_z_indices.shape)
# 使用广播机制生成新的索引网格
new_indices = torch.stack(torch.meshgrid(new_x_indices, new_y_indices, new_z_indices), dim=-1).view(-1, 3)
print(new_indices.shape)

net = torch.load('./trained_model_data_SS_precursor4D_field_data/1500.pth')
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
# np.save('trained_model_data_SS_precursor4D_field_data/1500.npy', predict)
truth_before_densify = np.load('trained_model_data_SS_precursor4D_field_data/1500.npy')

newdata = np.copy(predict)
for i in range(truth_before_densify.shape[1]):
    for j in range(truth_before_densify.shape[2]):
        for k in range(truth_before_densify.shape[3]):
            newdata[:, i*factor, j*factor, k*factor] = data[:, i, j, k]

plt.figure()
plt.imshow(data.reshape(201, 2800), aspect=2.5, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(ss4d_nmo.reshape(201, 2800), aspect=2.5, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
plt.xticks([])
plt.yticks([])
# plt.savefig('trained_model_data_SS_precursor4D_field_data/raw.pdf', dpi=300, bbox_inches='tight')

# plt.figure()
# plt.imshow(truth_before_densify.reshape(201, 2800), aspect=2.5, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('trained_model_data_SS_precursor4D_field_data/1500.pdf', dpi=300, bbox_inches='tight')

plt.figure()
plt.imshow(predict.reshape(201, data.shape[1]*factor*data.shape[2]*factor*data.shape[3]*factor), aspect=20, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
plt.xticks([])
plt.yticks([])
# plt.savefig('trained_model_data_SS_precursor4D_field_data/pro-1500_denoised_and_densify_201_40_32_140.pdf', dpi=300, bbox_inches='tight')
stack_recon = np.zeros((201, data.shape[1]*1, data.shape[2]*1), dtype=np.float32)
stack_raw = np.zeros((201, data.shape[1], data.shape[2]), dtype=np.float32)
for i in range(data.shape[1]*1):
    for j in range(data.shape[2]*1):
        # bot_col = data[:, i, j, [not np.all(data[:, i, j, k] == 0) for k in range(data.shape[3])]]
        # stack_raw[:, i, j] = np.mean(bot_col, axis=1)
        stack_recon[:, i, j] = np.mean(truth_before_densify[:, i, j, :], axis=1)
print(stack_recon.shape)
np.save('trained_model_data_SS_precursor4D_field_data/stackdata_1500.npy', stack_recon)
# plt.figure()
# plt.imshow(stack_recon.reshape(201, data.shape[1]*factor*data.shape[2]*factor), aspect=0.7, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
# plt.xticks([])
# plt.yticks([])

# plt.figure()
# plt.imshow(newdata.reshape(201, 22400)-predict.reshape(201, 22400), aspect=20, cmap=Seismics_map, vmin=-0.01, vmax=0.01)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.savefig('trained_model_data_SS_precursor3D_field_data/DRR_densify_SSprecursor3D_201_80_64_rank100_damped6-residual.pdf', dpi=300, bbox_inches='tight')
plt.show()