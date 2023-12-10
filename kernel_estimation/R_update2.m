
function R = R_update2(MSI_BS, HSI_2D, R, mu)
% MSI_BS 是 H ，Y*B
% HSI_2D 是 X_lr
% R初始是ones矩阵，进行归一化


mu = 1e-3;  % 设定一个值给 mu，通常这行不需要在函数参数中再定义一次

% 论文中将K=R
V = R;  
% 初始化 G 为与 R 相同大小的零矩阵，论文中的L，为拉格朗日算子
G = zeros(size(R)); 

% 计算 HSI_2D 乘以其转置的结果——X_lr X_lr^T
G1 = HSI_2D * HSI_2D';  

% 计算 MSI_BS 乘以 HSI_2D 的转置的结果--H X_lr ^T
G2 = MSI_BS * HSI_2D';  

for i = 1:100
    % 更新 R 的值
    R = (G2 + mu * V - G / 2) / (G1 + mu * eye(size(HSI_2D, 1)));  
    % 更新 V 的值,论文的K
    V = R + G / (2 * mu);  
    % 执行(·)_+操作，将小于0的置0
    V(V < 0) = 0;  

    % 更新 G 的值,更新L，拉格朗日算子
    G = G + 2 * mu * (R - V);  
end

