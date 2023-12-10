function s = csnr(A, B, row, col)
% CSNR计算函数
% A和B是两个图像或图像矩阵
% row和col是用于裁剪图像边界的行和列的数量

% 获取图像A的大小和通道数
[n, m, ch] = size(A);

% 初始化用于累加信噪比的变量
summa = 0;

% 如果图像只有一个通道（灰度图像）
if ch == 1
    % 计算两个图像的差异
    e = A - B;
    % 根据指定的行列参数裁剪图像边界
    e = e(row+1:n-row, col+1:m-col);
    % 计算图像中的最大像素值
    max_value = max(A(:));
    % 计算差异图像的均方误差
    me = mean(mean(e.^2));
    % 计算并返回改进的信噪比（CSNR）
    s = 10*log10(max_value^2 / me);
else
    % 对于多通道图像，循环处理每个通道
    for i = 1:ch
        % 计算两个图像的差异
        e = A - B;
        % 根据指定的行列参数裁剪图像边界
        e = e(row+1:n-row, col+1:m-col, i);
        % 计算差异图像的均方误差
        mse = mean(mean(e.^2));
        % 计算每个通道中的最大像素值
        max_value = max(max(max(A(:,:,i))));
        % 计算并返回每个通道的改进的信噪比（CSNR）
        s = 10*log10(max_value^2 / mse);
        % 对每个通道的CSNR进行累加
        summa = summa + s;
    end
    % 计算多通道图像的平均CSNR，并返回该值
    s = summa / ch;
end

% 函数结束
return;
