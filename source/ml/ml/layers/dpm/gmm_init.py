from typing import Tuple

import torch
from torch import Tensor

from ml.layers.clustering import KMeans, KMeans1D
from ml.layers.dpm.priors import Priors
from ml.utils import unique_count

GMMParams = Tuple[Tensor, Tensor, Tensor]

EPS = 0.0001


def initialize_kmeans(X: Tensor, k: int, prior: Priors, sim='euclidean') -> GMMParams:
    kmeans = KMeans(X.shape[1], k, sim=sim)
    kmeans.fit(X)

    mus = kmeans.get_centroids()

    I = kmeans.assign(X)
    covs = compute_covs_hard_assignment(X, I, k, mus, prior)

    counts = unique_count(I, k)
    pi = counts / counts.sum()

    mus = prior.compute_post_mus(counts, mus)
    covs = torch.stack([
        prior.compute_post_cov(counts[i], X[I == i].mean(axis=0), covs[i])
        for i in range(k)
    ])

    return mus, covs, pi


def initialize_kmeans1d(X: Tensor, k: int, prior: Priors, sim='euclidean') -> GMMParams:
    kmeans = KMeans1D(X.shape[1], k, sim=sim)
    kmeans.fit(X)

    mus = kmeans.get_centroids()
    I = kmeans.assign(X)
    covs = compute_covs_hard_assignment(X, I, k, mus, prior)

    counts = unique_count(I, k)
    pi = counts / counts.sum()

    mus = prior.compute_post_mus(counts, mus)
    covs = torch.stack([
        prior.compute_post_cov(counts[i], X[I == i].mean(axis=0), covs[i])
        for i in range(k)
    ])

    return mus, covs, pi


def initialize_soft_assignment(X: Tensor, r: Tensor, k: int, prior: Priors) -> GMMParams:
    N = r.shape[0]
    pi = r.sum(dim=0) / N

    mus = compute_mus_soft_assignment(X, r, k)
    covs = compute_covs_soft_assignment(X, r, k, mus)

    pi = prior.compute_post_pi(pi)
    mus = prior.compute_post_mus(pi * N, mus)

    covs = torch.stack([
        prior.compute_post_cov(pi[i] * len(X), mus[i], covs[i])
        for i in range(k)
    ])

    return mus, covs, pi


def update_cluster_params(X: Tensor, r: Tensor, k: int, prior: Priors) -> GMMParams:
    N = r.shape[0]
    pi = r.sum(dim=0) / N

    mus = compute_mus_soft_assignment(X, r, k)
    covs = compute_covs_soft_assignment(X, r, k, mus)

    pi = prior.compute_post_pi(pi)
    mus = prior.compute_post_mus(pi, mus)
    r_tot = r.sum(dim=0)
    covs = torch.stack([
        prior.compute_post_cov(r_tot[i], mus[i], covs[i])
        for i in range(k)
    ])

    return mus, covs, pi


def compute_covs_hard_assignment(X: Tensor, I: Tensor, k: int, mus: Tensor, prior: Priors):
    covs = []
    N = X.shape[0]
    for i in range(k):
        X_k = X[I == i]
        N_k = X_k.shape[0]
        if N_k > 0:
            cov_k = torch.matmul(
                (X_k - mus[i].repeat(N_k, 1)).T,
                (X_k - mus[i].repeat(N_k, 1)),
            ) / float(N_k)
        else:
            _, cov_k = prior.init_priors(X_k)
        covs.append(cov_k)
    return torch.stack(covs)


def compute_covs_soft_assignment(X: Tensor, r: Tensor, k: int, mus: Tensor):
    covs = []
    N = X.shape[0]
    N_k = r.sum(dim=0) + EPS
    for i in range(k):
        if len(X) > 0:
            cov_k = torch.matmul(
                (r[:, i] * (X - mus[i].repeat(N, 1)).T),
                (X - mus[i].repeat(N, 1)),
            ) / N_k[i]
        else:
            cov_k = torch.eye(mus.shape[1]) * EPS

        covs.append(cov_k)
    return torch.stack(covs)


def compute_mus_soft_assignment(X: Tensor, r: Tensor, k: int):
    denom = r.sum(dim=0)
    mus = torch.stack([
        (r[:, i].reshape(-1, 1) * X).sum(dim=0) / denom[i]
        for i in range(k)
    ]).detach()

    return mus