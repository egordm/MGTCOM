from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from ml.utils import unique_count, EPS, scatter_sum

MultivarNormalParams = NamedTuple("MultivarNormalParams", [("mus", Tensor), ("covs", Tensor)])

GMMParams = NamedTuple("MultivarNormalParamList", [("pis", Tensor), ("mus", Tensor), ("covs", Tensor)])


def compute_params_hard_assignment(X: Tensor, z: Tensor, k: int) -> Tuple[Tensor, MultivarNormalParams]:
    N_K = unique_count(z, k)

    mus = scatter_sum(X, z, k) / N_K.unsqueeze(1)
    covs = torch.stack([
        compute_cov(X[z == i], mus[i])
        for i in range(k)
    ])

    return N_K, MultivarNormalParams(mus, covs)


def compute_params_soft_assignment(X: Tensor, r: Tensor, k: int) -> Tuple[Tensor, MultivarNormalParams]:
    N_K = r.sum(dim=0) + EPS

    mus = torch.stack([
        (r[:, i].unsqueeze(1) * X).sum(dim=0) / N_K[i]
        for i in range(k)
    ])
    covs = torch.stack([
        compute_cov_soft(X, mus[i, :], r[:, i])
        for i in range(k)
    ])

    return N_K, MultivarNormalParams(mus, covs)


def compute_cov(X: Tensor, mu: Tensor):
    if len(X) > 0:
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(X_centered.T, X_centered) / float(X.shape[0])
    else:
        return torch.eye(mu.shape[-1]) * EPS


def compute_cov_soft(X: Tensor, mu: Tensor, r: Tensor):
    if len(X) > 0:
        N = r.sum(dim=0) + EPS
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(r * X_centered.T, X_centered) / N
    else:
        return torch.eye(mu.shape[-1]) * EPS


