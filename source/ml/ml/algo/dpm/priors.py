from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from ml.algo.dpm.statistics import MultivarNormalParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)

Float = Union[float, Tensor]

DirichletParams = NamedTuple('DirichletParams', [('alpha', Float)])


@dataclass
class DirichletPrior:
    params: DirichletParams

    @staticmethod
    def from_params(alpha: float = 1.0) -> Self:
        return DirichletPrior(DirichletParams(torch.tensor(alpha)))

    def compute_posterior(self, Ns: torch.Tensor) -> Tensor:
        return (Ns + self.params.alpha) / (Ns.sum() + self.params.alpha)


NIWPriorParams = NamedTuple('NIWPriorParams', [('nu', Float), ('kappa', Float), ('mu', Tensor), ('psi', Tensor)])
NIWPriorParamsList = NamedTuple('NIWPriorParamsList',
                                [('nus', Float), ('kappas', Float), ('mus', Tensor), ('psis', Tensor)])


@dataclass
class NIWPrior:
    params: NIWPriorParams

    @staticmethod
    def from_params(mu: Tensor, psi: Tensor, nu: Float = 12.0, kappa: Float = 0.0001) -> Self:
        D = mu.shape[-1]
        if nu < D:
            logger.warning("nu must be at least D + 1")
            nu = D + 1

        return NIWPrior(NIWPriorParams(
            torch.tensor(nu),
            torch.tensor(kappa),
            mu, psi,
        ))

    def update(self, X: Tensor, sigma_scale: float = 0.005):
        mu = X.mean(dim=0)
        psi = (torch.diag(X.std(dim=0)) * sigma_scale)

        self.params = NIWPriorParams(
            self.params.nu,
            self.params.kappa,
            mu, psi,
        )

    def compute_posterior(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> NIWPriorParamsList:
        # mu = X.mean(dim=0)
        # cov = compute_cov(X, mu)
        Ns = Ns.reshape(-1, 1)

        kappas_post = self.params.kappa + Ns
        nus_post = self.params.nu + Ns
        mus_post = (self.params.kappa * self.params.mu + Ns * mus) / kappas_post
        S = covs * Ns.unsqueeze(2)
        psis_post = (
                self.params.psi
                + S
                + ((self.params.kappa * Ns) / kappas_post).unsqueeze(2)
                * (mus - self.params.mu).unsqueeze(2) @ (mus - self.params.mu).unsqueeze(1)
        )

        return NIWPriorParamsList(nus_post, kappas_post, mus_post, psis_post)

    def compute_posterior_mv(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> MultivarNormalParams:
        D = mus.shape[-1]
        nus_post, kappas_post, mus_post, psis_post = self.compute_posterior(Ns, mus, covs)
        covs_post = torch.stack([
            psis_post[i] / (nus_post[i] - D + 1) if N_k > 0 else self.params.psi
            for i, N_k in enumerate(Ns)
        ], dim=0)

        return MultivarNormalParams(mus_post, covs_post)

    def marginal_log_prob(self, Ns: Tensor, mus: Tensor, covs: Tensor) -> Tensor:
        D = mus.shape[-1]
        nus_post, kappas_post, mus_post, psis_post = self.compute_posterior(Ns, mus, covs)

        # TODO: Compare
        return (
                -((Ns * D / 2) * np.log(torch.pi)).reshape(-1, 1)
                + torch.mvlgamma(nus_post / 2.0, D)
                - torch.mvlgamma(self.params.nu / 2.0, D)
                + (self.params.nu / 2.0) * torch.logdet(self.params.psi)
                - (nus_post / 2.0) * torch.logdet(psis_post).reshape(-1, 1)
                + (D / 2.0) * (torch.log(self.params.kappa) - torch.log(kappas_post))
        )
