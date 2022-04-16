from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from ml.utils import unique_count, EPS, scatter_sum

MultivarNormalParams = NamedTuple("MultivarNormalParams", [("mus", Tensor), ("covs", Tensor)])
DPMMParams = NamedTuple("DPMMParams", [("pis", Tensor), ("mus", Tensor), ("covs", Tensor)])
DPMMObs = NamedTuple("DPMMObs", [("Ns", Tensor), ("mus", Tensor), ("covs", Tensor)])


def compute_params_hard_assignment(X: Tensor, z: Tensor, k: int) -> DPMMObs:
    Ns = unique_count(z, k)

    mus = scatter_sum(X, z, k) / Ns.unsqueeze(1)
    covs = torch.stack([
        compute_cov(X[z == i], mus[i])
        for i in range(k)
    ])

    return DPMMObs(Ns, mus, covs)


def compute_params_soft_assignment(X: Tensor, r: Tensor, k: int) -> DPMMObs:
    Ns = r.sum(dim=0) + EPS

    mus = torch.stack([
        (r[:, i].unsqueeze(1) * X).sum(dim=0) / Ns[i]
        for i in range(k)
    ])
    covs = torch.stack([
        compute_cov_soft(X, mus[i, :], r[:, i])
        for i in range(k)
    ])

    return DPMMObs(Ns, mus, covs)


def compute_cov(X: Tensor, mu: Tensor) -> Tensor:
    if len(X) > 0:
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(X_centered.T, X_centered) / float(X.shape[0])
    else:
        return torch.eye(mu.shape[-1]) * EPS


def compute_cov_soft(X: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    if len(X) > 0:
        N = r.sum(dim=0) + EPS
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(r * X_centered.T, X_centered) / N
    else:
        return torch.eye(mu.shape[-1]) * EPS
