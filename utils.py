import numpy as np
from scipy import signal
from scipy.io import loadmat
import scipy.io as sio
import tifffile
import sys
import h5py


def warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr = init_lr1 + iteraion / warm_iter * (init_lr2 - init_lr1)
    else:
        lr = init_lr2 * (1 - (iteraion - warm_iter) / (max_iter - warm_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


def Gaussian_downsample(x, psf, s):
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate(optimizer, epoch, milestones=None, warmup=0, base_lr=0.001):
    def to(epoch):
        if epoch <= warmup:
            return 1
        elif warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_F():
    F = np.array([[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
                  [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
                  [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


def dataset_input(data_put, downsample_factor):

    if data_put == 'pavia':
        F = loadmat('./data/R.mat')
        F = F['R']
        F = F[:, 0:-10]
        for band in range(F.shape[0]):
            div = np.sum(F[band][:])
            for i in range(F.shape[1]):
                # L1 归一化
                F[band][i] = F[band][i] / div
        R = F
        HRHSI = tifffile.imread('./data/original_rosis.tif')
        HRHSI = HRHSI[0:-10, 0:downsample_factor ** 2 * int(HRHSI.shape[1] / downsample_factor ** 2), 0:downsample_factor ** 2 * int(HRHSI.shape[2] / downsample_factor ** 2)]
        HRHSI = HRHSI / np.max(HRHSI)
    elif data_put == 'Chikusei':
        mat = h5py.File('.\data\Chikusei.mat')
        HRHSI = mat['chikusei']
        mat1 = sio.loadmat('.\data\Chikusei_data.mat')
        R = mat1['R']
        R = R[0:8:2, :]
        HRHSI = HRHSI[:, 100:900, 100:900]
        HRHSI = np.transpose(HRHSI, (0, 2, 1))
        x1 = np.max(HRHSI)
        x2 = np.min(HRHSI)
        x3 = -x2 / (x1 - x2)
        HRHSI = HRHSI / (x1 - x2) + x3
    elif data_put == 'houston':
        mat = sio.loadmat('.\data\Houston.mat')
        HRHSI = mat['Houston']
        HRHSI = np.transpose(HRHSI, (2, 0, 1))
        HRHSI = HRHSI[:, 0:336, 100:900]
        x1 = np.max(HRHSI)
        x2 = np.min(HRHSI)
        x3 = -x2 / (x1 - x2)
        HRHSI = HRHSI / (x1 - x2) + x3
        R = np.zeros((4, HRHSI.shape[0]))
        for i in range(R.shape[0]):
            R[i, 36 * i:36 * (i + 1)] = 1 / 36.0
    else:
        sys.exit(0)
    return HRHSI, R

# 保存生成的训练数据
# def savedata(dataset, R, training_size, stride, downsample_factor, PSF, num):
#     if dataset == 'CAVE':
#         path = 'D:\我的代码\高光谱集数据\CAVE\\'
#     elif dataset == 'Harvard':
#         path = 'D:\我的代码\高光谱集数据\Harvard\\'
#     imglist = os.listdir(path)
#     train_hrhs = []
#     train_hrms = []
#     train_lrhs = []
#     for i in range(num):
#         img = loadmat(path + imglist[i])
#         img1 = img["b"]
#         HRHSI = np.transpose(img1, (2, 0, 1))
#         HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
#         MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
#         for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
#             for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
#                 temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
#                 temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
#                 temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor), int(k / downsample_factor):int((k + training_size) / downsample_factor)]
#                 train_hrhs.append(temp_hrhs)
#                 train_hrms.append(temp_hrms)
#                 train_lrhs.append(temp_lrhs)
#         sio.savemat(dataset + '.mat', {'hrhs': train_hrhs, 'ms': train_hrms, 'lrhs': train_lrhs})
#         print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)
