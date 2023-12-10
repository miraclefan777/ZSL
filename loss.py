import torch
import torch.nn.functional as functional
import torch.nn as nn


class MyarcLoss(nn.Module):
    def __init__(self, R, PSF):
        super(MyarcLoss, self).__init__()
        self.R = R
        self.PSF = PSF

    def forward(self, output, target, MSI, sf):
        coeff = torch.unsqueeze(self.PSF, 0)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.repeat_interleave(coeff, output.shape[1], 0)
        _, c, h, w = output.shape
        w1, h1 = self.PSF.shape
        outs = functional.conv2d(output, coeff.cuda(), bias=None, stride=sf, padding=int((w1 - 1) / 2), groups=c)
        target_HSI = functional.conv2d(target, coeff.cuda(), bias=None, stride=sf, padding=int((w1 - 1) / 2), groups=c)
        RTZ = torch.tensordot(output, self.R, dims=([1], [1]))
        RTZ = torch.Tensor.permute(RTZ, (0, 3, 1, 2))
        MSILoss = torch.mean(torch.abs(RTZ - MSI))
        tragetloss = torch.mean(torch.abs(output - target))
        HSILoss = torch.mean(torch.abs(outs[:, :, 1:-1, 1:-1] - target_HSI[:, :, 1:-1, 1:-1]))
        loss_total = MSILoss + tragetloss + 0.1 * HSILoss
        return loss_total
