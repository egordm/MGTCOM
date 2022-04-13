import torch
from torch import Tensor


def compute_cov(X: Tensor):
    return torch.matmul(X.T, X) / float(X.shape[0])