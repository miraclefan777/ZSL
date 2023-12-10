# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
import math
import os
import numpy as np
import cv2
from tqdm import tqdm

from loss import MyarcLoss
from model import CNN
from dataset import HSI_MSI_Data
from utils import warm_lr_scheduler, fspecial, Gaussian_downsample, dataset_input
import metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PSF = fspecial('gaussian', 7, 3)
p = 10
stride = 1
training_size = 32
downsample_factor = 4
LR = 1e-3
EPOCH = 400
BATCH_SIZE = 64
loss_optimal = 1.75
init_lr1 = 1e-4
init_lr2 = 5e-4
decay_power = 1.5
data2 = 'pavia'

"""
模拟数据生成，假设已知psf和R，使用dataset_input
对于真实情况，如果是已知HR-MSI和LR-HSI，可以通过调用Kernal_estimation.m近似求出R和psf
然后将R，psf代入到下面的式子中直接进入子空间表示，降维部分，HSI0=LR-HSI

接着对HSI0进行数据增强，然后在生成数据过程中，采样得到X_d,对于Y_d则是通过响应函数直接计算得到。
"""
[HRHSI, R] = dataset_input(data2, downsample_factor)

maxiteration = (2 * math.ceil(
    ((HRHSI.shape[1] / downsample_factor - training_size) // stride + 1) *
    ((HRHSI.shape[2] / downsample_factor - training_size) // stride + 1)
    / BATCH_SIZE)
                * EPOCH)
warm_iter = math.floor(maxiteration / 40)
print(maxiteration)

"""计算X_lr"""
# (X*B)D
HSI0 = Gaussian_downsample(HRHSI, PSF, downsample_factor)
# (X*B)D+N_x=X_lr
HSI0 = HSI0 + 0 * np.random.randn(HSI0.shape[0], HSI0.shape[1], HSI0.shape[2])

"""计算y，用来给model提供channel数据"""
MSI0 = np.tensordot(R, HRHSI, axes=([1], [0]))

"""子空间表示，降维"""
HSI3 = HSI0.reshape(HSI0.shape[0], -1)
# 奇异值分解用来降维
U0, S, V = np.linalg.svd(np.dot(HSI3, HSI3.T))
# p为论文中的L，该奇异值分解保留了L个最大奇异值
U0 = U0[:, 0:int(p)]
# 论文中的A_lr
HSI0_Abun = np.tensordot(U0.T, HSI0, axes=([1], [0]))

augument = [0]
HSI_aug = []
HSI_aug.append(HSI0)
U = U0

"""生成训练数据"""
train_hrhs = []
train_hrms = []
train_lrhs = []
for j in augument:
    # 垂直翻转
    HSI = cv2.flip(HSI0, j)
    # MSI_aug.append(MSI0)
    HSI_aug.append(HSI)

for j in range(len(HSI_aug)):
    HSI = HSI_aug[j]

    """计算A_d"""
    HSI_Abun = np.tensordot(U.T, HSI, axes=([1], [0]))
    HSI_LR_Abun = Gaussian_downsample(HSI_Abun, PSF, downsample_factor)

    """计算Y_d"""
    MSI_LR = np.tensordot(R, HSI, axes=([1], [0]))

    for j in range(0, HSI_Abun.shape[1] - training_size + 1, stride):
        for k in range(0, HSI_Abun.shape[2] - training_size + 1, stride):
            temp_hrhs = HSI[:, j:j + training_size, k:k + training_size]
            temp_hrms = MSI_LR[:, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR_Abun[:, int(j / downsample_factor):int((j + training_size) / downsample_factor), int(k / downsample_factor):int((k + training_size) / downsample_factor)]

            train_hrhs.append(temp_hrhs)
            train_hrms.append(temp_hrms)
            train_lrhs.append(temp_lrhs)

train_hrhs = torch.Tensor(np.array(train_hrhs))
train_lrhs = torch.Tensor(np.array(train_lrhs))
train_hrms = torch.Tensor(np.array(train_hrms))

print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)

train_data = HSI_MSI_Data(train_hrhs, train_hrms, train_lrhs)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

"""
定义模型相关参数、优化器、损失
"""
cnn = CNN(p, MSI0.shape[0]).cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
PSF_T = torch.Tensor(PSF)
loss_func = MyarcLoss(torch.Tensor(R).cuda(), PSF_T.cuda())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2, last_epoch=-1)

# 初始化模型参数
for m in cnn.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

MSI_1 = torch.Tensor(np.expand_dims(MSI0, axis=0))
HSI_1 = torch.Tensor(np.expand_dims(HSI0_Abun, axis=0))
step = 0
loss_list = []
U22 = torch.Tensor(U0)
for epoch in range(EPOCH):
    loop = tqdm(train_loader, total=len(train_loader))
    loss_all = []
    for (a1, a2, a3) in loop:
        cnn.train()

        lr = warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, step, lr_decay_iter=1, max_iter=maxiteration, power=decay_power)
        step = step + 1
        output = cnn(a3.cuda(), a2.cuda())
        Fuse1 = torch.tensordot(U22.cuda(), output, dims=([1], [1]))
        Fuse1 = torch.Tensor.permute(Fuse1, (1, 0, 2, 3))
        loss = loss_func(Fuse1, a1.cuda(), a2.cuda(), downsample_factor)

        loss_temp = loss
        loss_all.append(np.array(loss_temp.detach().cpu().numpy()))

        # 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
        loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})

    cnn.eval()
    with torch.no_grad():
        abudance = cnn(HSI_1.cuda(), MSI_1.cuda())
        abudance = abudance.cpu().detach().numpy()
        abudance1 = np.squeeze(abudance)
    Fuse2 = np.tensordot(U0, abudance1, axes=([1], [0]))
    sum_loss, psnr_ = metrics.rmse1(np.clip(Fuse2, 0, 1), HRHSI)
    if sum_loss < loss_optimal:
        loss_optimal = sum_loss
    loss_list.append(sum_loss)
    print("Epoch:{0},SUM_Loss:{1},PSNR:{2}".format(epoch, sum_loss, psnr_))

    # 5个epoch保存一次
    if epoch % 5 == 0:
        torch.save(cnn.state_dict(), data2 + str(epoch) + '.pkl', _use_new_zipfile_serialization=False)
