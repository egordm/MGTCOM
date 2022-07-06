from enum import Enum
from typing import NamedTuple

import torch
from torch import Tensor

from ml.algo.clustering import KMeans, KMeans1D
from ml.utils import Metric

EPS = 1e-6

GaussianParams = NamedTuple('GaussianParams', [('Ns', Tensor), ('mus', Tensor), ('covs', Tensor)])


def estimate_gaussian_parameters(X: Tensor, r: Tensor, reg_covar: float = 1e-6) -> GaussianParams:
    Ns = r.sum(dim=0) + EPS
    mus = torch.mm(r.T, X) / Ns[:, None]

    k, D = mus.shape
    covs = torch.zeros(k, D, D)
    for i in range(k):
        diff = X - mus[i][None, :]
        covs[i] = torch.mm(r[:, i] * diff.T, diff) / Ns[i] + torch.eye(D) * reg_covar

    return GaussianParams(Ns, mus, covs)


def covs_to_prec(covs):
    cov_chol = torch.linalg.cholesky(covs)
    Id = torch.eye(covs.shape[-1], dtype=covs.dtype, device=covs.device)
    prec_chol = torch.linalg.solve_triangular(cov_chol, Id, upper=False)
    return prec_chol.transpose(-1, -2)


def estimate_gaussian_log_prob(X: Tensor, mus: Tensor, precs: Tensor) -> Tensor:
    k, D = mus.shape
    ys = torch.bmm(X.expand(k, *X.shape), precs) - torch.bmm(mus.unsqueeze(1), precs)
    M = ys.square().sum(dim=-1).T

    half_log_det = precs.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    return -0.5 * (D * torch.log(2 * torch.tensor(torch.pi)) + M) + half_log_det.unsqueeze(0)


class InitMode(Enum):
    RANDOM = 'random'
    KMEANS = 'kmeans'
    KMEANS1D = 'kmeans1d'
    HARD = 'hard'


def initial_assignment(X: Tensor, k: int, mode: InitMode, metric: Metric, z_init: Tensor = None) -> Tensor:
    N, D = X.shape

    if z_init is not None:
        r = torch.zeros(N, k)
        r[torch.arange(N), z_init] = 1
    elif mode == InitMode.KMEANS:
        z = KMeans(D, k, metric).fit(X).assign(X)
        r = torch.zeros(N, k)
        r[torch.arange(N), z] = 1
    elif mode == InitMode.KMEANS1D:
        z = KMeans1D(D, k, metric).fit(X).assign(X)
        r = torch.zeros(N, k)
        r[torch.arange(N), z] = 1
    else:
        r = torch.rand(N, k)
        r /= r.sum(dim=1, keepdim=True)

    return r


def to_hard_assignment(log_r: Tensor) -> Tensor:
    z = log_r.argmax(dim=1)
    r = torch.zeros_like(log_r)
    r[torch.arange(len(log_r)), z] = 1
    return r

def merge_params(Ns: Tensor, mus: Tensor, covs: Tensor) -> GaussianParams:
    mus = mus[Ns > 0]
    covs = covs[Ns > 0]
    Ns = Ns[Ns > 0]

    Ns_c = Ns.sum(dim=0, keepdim=True)

    if len(mus) == 1:
        return GaussianParams(Ns_c, mus, covs)
    elif len(mus) > 1:  # Weighted mean
        mus_c = (Ns.reshape(-1, 1) * mus).sum(dim=0, keepdim=True) / Ns_c.reshape(-1, 1)
        covs_c = (
            (
                Ns.reshape(-1, 1, 1)
                * (covs + mus.unsqueeze(2) @ mus.unsqueeze(1))
            ).sum(dim=0, keepdim=True)
            / Ns_c.reshape(-1, 1, 1)
            - mus_c.unsqueeze(2) @ mus_c.unsqueeze(1)
        )
        return GaussianParams(Ns_c, mus_c, covs_c)
    else:
        raise ValueError('Trying to merge empty clusters')
