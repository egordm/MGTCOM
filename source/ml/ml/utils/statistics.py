import torch
from torch import Tensor

EPS = 0.0001


def norm_eps(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


def compute_cov_old(X: Tensor):
    return torch.matmul(X.T, X) / float(X.shape[0])


def compute_cov(X: Tensor, mu: Tensor):
    X_centered = X - mu.unsqueeze(0)
    return torch.matmul(X_centered.T, X_centered) / float(X.shape[0])


def compute_cov_soft(X: Tensor, mu: Tensor, r: Tensor):
    if len(X) > 0:
        N = r.sum() + EPS
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(r * X_centered.T, X_centered) / N
    else:
        return torch.eye(mu.shape[-1]) * EPS
