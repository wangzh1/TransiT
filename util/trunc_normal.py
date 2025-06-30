import torch
from scipy.stats import truncnorm

def trunc_normal_(tensor, mean=0, std=1):
    # 计算截断正态分布的边界
    lower, upper = -2 * std, 2 * std
    # 产生一个截断正态分布
    samples = truncnorm.rvs(
        (lower - mean) / std, (upper - mean) / std, scale=std, size=tensor.shape
    )
    # 将样本转化为张量
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(samples))
    return tensor

