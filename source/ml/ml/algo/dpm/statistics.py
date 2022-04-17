from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from ml.utils import unique_count, EPS, scatter_sum

MultivarNormalParams = NamedTuple("MultivarNormalParams", [("mus", Tensor), ("covs", Tensor)])
DPMMParams = NamedTuple("DPMMParams", [("pis", Tensor), ("mus", Tensor), ("covs", Tensor)])
DPMMObs = NamedTuple("DPMMObs", [("Ns", Tensor), ("mus", Tensor), ("covs", Tensor)])


def compute_params_hard_assignment(X: Tensor, z: Tensor, k: int) -> DPMMObs:
    """
    It computes the parameters of a DPMM given the data and the hard assignments

    :param X: the data
    :type X: Tensor
    :param z: the cluster assignments
    :type z: Tensor
    :param k: number of clusters
    :type k: int
    :return: The number of points in each cluster, the mean of each cluster, and the covariance matrix of each cluster.
    """
    Ns = unique_count(z, k)

    mus = scatter_sum(X, z, k) / Ns.unsqueeze(1)
    covs = torch.stack([
        compute_cov(X[z == i], mus[i])
        for i in range(k)
    ])

    return DPMMObs(Ns, mus, covs)


def compute_params_soft_assignment(X: Tensor, r: Tensor, k: int) -> DPMMObs:
    """
    It computes the parameters of the mixture model by taking the weighted average of the data points

    :param X: the data
    :type X: Tensor
    :param r: the soft assignment of each data point to each cluster
    :type r: Tensor
    :param k: number of clusters
    :type k: int
    :return: The number of points in each cluster, the mean of each cluster, and the covariance matrix of each cluster.
    """
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
    """
    It computes the covariance matrix of a set of points, given the mean of those points

    :param X: the data
    :type X: Tensor
    :param mu: the mean of the data
    :type mu: Tensor
    :return: The covariance matrix of the data.
    """
    if len(X) > 0:
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(X_centered.T, X_centered) / float(X.shape[0])
    else:
        return torch.eye(mu.shape[-1]) * EPS


def compute_cov_soft(X: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """
    It computes the covariance matrix of a set of points, weighted by a set of weights

    :param X: the data
    :type X: Tensor
    :param mu: the mean of the Gaussian
    :type mu: Tensor
    :param r: the responsibilities of each data point to each cluster
    :type r: Tensor
    :return: The covariance matrix of the data.
    """
    if len(X) > 0:
        N = r.sum(dim=0) + EPS
        X_centered = X - mu.unsqueeze(0)
        return torch.matmul(r * X_centered.T, X_centered) / N
    else:
        return torch.eye(mu.shape[-1]) * EPS


def merge_params(Ns: Tensor, mus: Tensor, covs: Tensor) -> DPMMObs:
    """
    It takes the parameters of a bunch of Gaussians and returns the parameters of a single Gaussian that is the weighted
    mean of the input Gaussians

    :param Ns: The number of observations in each cluster
    :type Ns: Tensor
    :param mus: The mean of each cluster
    :type mus: Tensor
    :param covs: Tensor of shape (K, D, D)
    :type covs: Tensor
    :return: The number of observations, the mean of the observations, and the covariance of the observations.
    """
    Ns_c = Ns.sum(dim=0, keepdim=True)

    if Ns_c > 0:  # Weighted mean
        mus_c = (Ns.reshape(-1, 1) * mus).sum(dim=0, keepdim=True) / Ns_c.reshape(-1, 1)
        covs_c = (
                (
                        Ns.reshape(-1, 1, 1)
                        * (covs + mus.unsqueeze(2) @ mus.unsqueeze(1))
                ).sum(dim=0, keepdim=True)
                / Ns_c.reshape(-1, 1, 1)
                - mus_c.unsqueeze(2) @ mus_c.unsqueeze(1)
        )
    else:
        mus_c = mus.mean(dim=0, keepdim=True)
        covs_c = covs.mean(dim=0, keepdim=True)

    return DPMMObs(Ns_c, mus_c, covs_c)
