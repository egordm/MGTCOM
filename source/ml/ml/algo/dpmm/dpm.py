from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from ml.algo.dpmm.base import BaseMixture, MixtureParams, P
from ml.algo.dpmm.prior import DirPrior, NWPrior, DirParams, NWParams
from ml.algo.dpmm.statistics import estimate_gaussian_parameters, initial_assignment
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class DPMMParams(NamedTuple):
    dir: DirParams
    nw: NWParams


@dataclass
class DPMixtureParams(MixtureParams):
    prior_alpha: float = None
    prior_kappa: float = 1.0
    prior_nu: float = None
    prior_sigma_scale: float = 1.0


class DirichletProcessMixture(BaseMixture[DPMMParams]):
    hparams: DPMixtureParams
    prior_dir: DirPrior = None
    prior_nw: NWPrior = None

    def __init__(self, hparams: DPMixtureParams) -> None:
        super().__init__(hparams)

    def _init(self, X: Tensor) -> None:
        self.hparams.prior_alpha = 1.0 / self.n_components if self.hparams.prior_alpha is None \
            else self.hparams.prior_alpha
        self.prior_dir = DirPrior.from_params(self.hparams.prior_alpha)

        _, D = X.shape
        if self.hparams.prior_nu is None:
            self.hparams.prior_nu = D + 1
        elif self.hparams.prior_nu <= D:
            logger.warning(f"nu must be at least D + 1. Setting nu to {D + 1}")
            self.hparams.prior_nu = D + 1

        mu, cov = torch.mean(X, dim=0), torch.cov(X.T)
        self.prior_nw = NWPrior.from_params(
            self.hparams.prior_kappa, self.hparams.prior_nu, mu, cov, self.hparams.prior_sigma_scale
        )

    def _init_params(self, X: Tensor) -> None:
        r = initial_assignment(X, self.n_components, self.hparams.init_mode, self.hparams.metric)
        Ns, mus, covs = estimate_gaussian_parameters(X, r, self.hparams.reg_cov)

        Ns_post = self.prior_dir.estimate_post(Ns)
        params_post = self.prior_nw.estimate_post(Ns, mus, covs)
        self._set_params(DPMMParams(Ns_post, params_post))

    def _m_step(self, X: Tensor, log_r: Tensor):
        Ns, mus, covs = estimate_gaussian_parameters(X, log_r.exp(), self.hparams.reg_cov)

        Ns_post = self.prior_dir.estimate_post(Ns)
        params_post = self.prior_nw.estimate_post(Ns, mus, covs)
        self._set_params(DPMMParams(Ns_post, params_post))

    def _estimate_log_weights(self) -> Tensor:
        return self.prior_dir.estimate_log_prob(self.params.dir)

    def _estimate_log_prob(self, X: Tensor) -> Tensor:
        return self.prior_nw.estimate_log_prob(X, self.params.nw)

    def _compute_lower_bound(self, X, log_r) -> Tensor:
        _, D = self.params.nw.mus.shape
        log_wishart = self.prior_nw.log_norm(self.params.nw.nus, self.params.nw.Ws, D).sum()
        log_dir = -self.prior_dir.log_norm(self.params.dir).sum()

        return (
            -torch.sum(log_r.exp() * log_r)
            - log_wishart
            - log_dir
            - 0.5 * D * self.params.nw.kappas.log().sum()
        )

    def _set_params(self, params: DPMMParams) -> None:
        super()._set_params(params)
        self.n_components = len(params.nw.mus)

    @property
    def mus(self) -> Tensor:
        return self.params.nw.mus

    @property
    def covs(self) -> Tensor:
        return self.params.nw.covs
