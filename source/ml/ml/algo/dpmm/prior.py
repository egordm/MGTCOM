import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, lgamma, mvlgamma, digamma
from typing_extensions import Self

from ml.algo.dpmm.statistics import estimate_gaussian_log_prob, covs_to_prec
from ml.utils import batchwise_outer
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class DirParams(NamedTuple):
    a: Tensor
    b: Tensor


@dataclass
class DirPrior:
    """
    Dirichlet prior for multinomial distribution.
    """

    alpha: Tensor
    """Concentration parameter of the Dirichlet distribution."""

    @staticmethod
    def from_params(alpha: float) -> Self:
        return DirPrior(torch.tensor(alpha))

    @staticmethod
    def log_norm(params: DirParams) -> Tensor:
        return lgamma(params.a) + lgamma(params.b) - lgamma(params.a + params.b)

    def estimate_post(self, Ns: Tensor) -> DirParams:
        """
        Estimate concentration parameter of the Dirichlet distribution.
        """
        return DirParams(
            1.0 + Ns,
            self.alpha + torch.tensor(np.hstack([np.cumsum(Ns.numpy()[::-1])[-2::-1], 0.0]), dtype=torch.float)
        )

    @staticmethod
    def estimate_log_prob(params: DirParams) -> Tensor:
        # Basically computes log pi
        digamma_sum = torch.digamma(params.a + params.b)
        digamma_a = torch.digamma(params.a)
        digamma_b = torch.digamma(params.b)
        return (
            digamma_a - digamma_sum +
            torch.hstack((torch.tensor(0.0), torch.cumsum(digamma_b - digamma_sum, dim=0)[:-1]))
        )


class NWParams(NamedTuple):
    mus: Tensor
    kappas: Tensor
    nus: Tensor
    Ws: Tensor
    covs: Tensor

    def __getitem__(self, item):
        return NWParams(self.mus[item], self.kappas[item], self.nus[item], self.Ws[item], self.covs[item])


@dataclass
class NWPrior:
    """
    Normal Wishart prior for Gaussian distribution.
    """
    mu_0: Tensor
    """Mean of the Gaussian distribution."""
    kappa: Tensor
    """Concentration parameter of the Wishart distribution."""
    nu: Tensor
    """Degrees of freedom of the Wishart distribution."""
    W_inv: Tensor
    """Inverse of the scale matrix of the Wishart distribution. (covariance matrix)"""

    @staticmethod
    def from_data(X: Tensor, kappa: float, nu: float, prior_cov_scale: float = 1.0) -> Self:
        assert prior_cov_scale > 0, 'prior_cov_scale must be positive'

        mu, cov = torch.mean(X, dim=0), torch.diag(torch.std(X, dim=0))  # torch.cov(X.T)
        return NWPrior.from_params(
            kappa, nu, mu, cov * prior_cov_scale
        )

    @staticmethod
    def from_params(kappa: float, nu: float, mu: Tensor, cov: Tensor) -> Self:
        assert kappa > 0, 'kappa must be positive'
        assert nu >= mu.shape[0] + 1, 'nu must be larger or equal to D + 1'

        return NWPrior(mu, torch.tensor(kappa), torch.tensor(nu), cov)

    @staticmethod
    def log_norm(nu: Tensor, W: Tensor, D: int) -> Tensor:
        return -(
            (W.logdet() - 0.5 * D * nu.log()) * nu
            + np.log(2) * (nu * D / 2)
            + mvlgamma(nu / 2, D)
        )

    def estimate_post(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> NWParams:
        _, D = mus.shape
        kappas_k = self.kappa + Ns
        nus_k = self.nu + Ns
        mus_k = (self.kappa * self.mu_0 + Ns[:, None] * mus) / kappas_k[:, None]

        diff = mus_k - self.mu_0[None, :]
        covs_k = (
                     self.W_inv[None, :, :]
                     + Ns[:, None, None] * covs
                     + ((self.kappa * Ns) / kappas_k)[:, None, None]
                     * batchwise_outer(diff, diff)
                 ) / (nus_k[:, None, None] + D + 2) # TODO: check whether D + 2 is fine
        Ws_k = covs_to_prec(covs_k)

        return NWParams(mus_k, kappas_k, nus_k, Ws_k, covs_k)

    @staticmethod
    def estimate_log_prob(X: Tensor, params: NWParams) -> Tensor:
        # Basically Multi-variate Normal Distribution with computed mu and cov (or W in this case which is its inverse)
        k, D = params.mus.shape
        log_gauss = estimate_gaussian_log_prob(X, params.mus, params.Ws) - 0.5 * D * params.nus.log()
        log_lambda = (
            D * math.log(2.0)
            + digamma(0.5 * (params.nus - torch.arange(D)[:, None])).sum(dim=0)
        )  # Bishop eq. (B.81)

        return log_gauss + 0.5 * (log_lambda - D / params.kappas)  # Bishop eq. (B.78)

    def estimate_marginal_log_prob(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> Tensor:
        # Computes: P(D_new | D)
        # Note: Use hard assignment with this one
        _, D = mus.shape
        mus_post, kappas_post, nus_post, Ws_post, covs_post = self.estimate_post(Ns, mus, covs)

        return (
            -(np.log(torch.pi) * (Ns * D / 2.0))
            + mvlgamma(nus_post / 2.0, D)
            - mvlgamma(self.nu / 2.0, D)
            + self.W_inv.logdet() * (self.nu / 2.0)
            - (covs_post * (nus_post[:, None, None] + D + 2)).logdet() * (nus_post / 2.0) # TODO: shouldn't we use Ws instead?
            + (torch.log(self.kappa) - torch.log(kappas_post)) * (D / 2.0)
        )

# class NIWPrior:
#     """
#     Normal Inverse Wishart prior for Gaussian distribution.
#     """
#     mu_0: Tensor
#     """Mean of the Gaussian distribution."""
#     kappa: Tensor
#     """Concentration parameter of the Wishart distribution."""
#     nu: Tensor
#     """Degrees of freedom of the Wishart distribution."""
#     psi: Tensor
#     """Inverse Scale matrix of the Wishart distribution."""
#
#     @staticmethod
#     def log_norm(nu: Tensor, psi: Tensor, D: int) -> Tensor:
#         return (
#             - psi.logdet() * (nu / 2)
#             - math.log(2) * (nu * D / 2)
#             - mvlgamma(nu / 2, D)
#         )
