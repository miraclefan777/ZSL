import torch
import torch.utils.data as data
import h5py


class HSI_MSI_Data(data.Dataset):
    def __init__(self, train_hrhs_all, train_hrms_all, train_lrhs_all):
        self.train_hrhs_all = train_hrhs_all
        self.train_hrms_all = train_hrms_all
        self.train_lrhs_all = train_lrhs_all

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class HSI_MSI_Data1(data.Dataset):
    def __init__(self, dataset):
        mat = h5py.File(dataset + '.mat')
        self.train_hrhs_all = mat['hrhs']
        self.train_hrms_all = mat['ms']
        self.train_lrhs_all = mat['lrhs']

    def __getitem__(self, index):
        train_hrhs = torch.Tensor(self.train_hrhs_all[index, :, :, :])
        train_hrms = torch.Tensor(self.train_hrms_all[index, :, :, :])
        train_lrhs = torch.Tensor(self.train_lrhs_all[index, :, :, :])
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]
