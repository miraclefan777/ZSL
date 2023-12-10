function [R, B] = Kernal_estimation(HSI, MSI, size_B)
% HSI: 高光谱图像
% MSI: 多光谱图像
% size_B: 核的大小

[M, N, L] = size(MSI);  % 获取多光谱图像的尺寸信息
R = ones(size(MSI, 3), size(HSI, 3));  % 初始化核估计矩阵 R
R = R / size(HSI, 3);  

% 将多光谱图像和高光谱图像转换为2D格式
HSI_2D = hyperConvert2D(HSI);
% MSI_2D = hyperConvert2D(MSI);

s0 = 1;
sf = size(MSI, 1) / size(HSI, 1);  % 计算下采样因子
iter = 20;  % 迭代次数
% size_B = 7;  % 核大小

% 初始化核 B
B = abs(randn(size_B, size_B));
mu = 0;
B = B / sum(B(:));  % 对核进行归一化

% 开始迭代
for i = 1:iter
    % 计算核的频域表示
    fft_B = psf2otf(B, [M N]);
    % 对多光谱图像进行高斯下采样
    MSI_BS = Gaussian_downsample(MSI, fft_B, sf, s0);
    MSI_BS = hyperConvert2D(MSI_BS);
    % 更新核估计矩阵 R，式子（6，9）
    R = R_update2(MSI_BS, HSI_2D, R, mu);

    % 根据更新后的 R 重构高光谱图像
    R_HSI = R * HSI_2D;
    R_HSI = hyperConvert3D(R_HSI, M/sf, N/sf);
    R_HSI_up = zeros(M, N, L);
    R_HSI_up(s0:sf:end, s0:sf:end, :) = R_HSI;

    % 更新核 B
    B = B_update2(R_HSI_up, MSI, size_B, sf, B, mu);
end
end

