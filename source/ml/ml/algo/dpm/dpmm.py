from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from ml.algo.clustering import KMeans, KMeans1D
from ml.algo.dpm.mhmc import MHMC
from ml.algo.dpm.statistics import compute_params_hard_assignment, compute_params_soft_assignment, DPMMObs
from ml.utils import Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class InitMode(Enum):
    KMeans = 'kmeans'
    KMeans1D = 'kmeans1d'
    SoftAssignment = 'soft_assignment'


class DirichletProcessMixtureModel(torch.nn.Module):
    mhmc: MHMC
    components: List[MultivariateNormal] = None

    def __init__(self, n_components: int, repr_dim: int, metric: Metric, mhmc: MHMC):
        super().__init__()
        self.n_components = n_components
        self.repr_dim = repr_dim
        self.metric = metric
        self.mhmc = mhmc

        self._pis = torch.nn.Parameter(torch.ones(self.n_components) / self.n_components, requires_grad=False)
        self._mus = torch.nn.Parameter(torch.randn(self.n_components, self.repr_dim), requires_grad=False)
        self._covs = torch.nn.Parameter(torch.eye(self.repr_dim).unsqueeze(0).repeat(self.n_components, 1, 1),
                                        requires_grad=False)

    @property
    def is_initialized(self) -> bool:
        return self.components is not None

    def _set_params(self, pis: Tensor, mus: Tensor, covs: Tensor):
        self.n_components = len(pis)
        assert pis.shape == (self.n_components,)
        assert mus.shape == (self.n_components, self.repr_dim)
        assert covs.shape == (self.n_components, self.repr_dim, self.repr_dim)

        self._pis.data, self._mus.data, self._covs.data = pis, mus, covs
        self.components = [MultivariateNormal(mu, cov) for mu, cov in zip(mus, covs)]

    def reinitialize(self, X: Tensor, r: Optional[Tensor], mode: InitMode = InitMode.KMeans):
        if mode == InitMode.SoftAssignment:
            obs = compute_params_soft_assignment(X, r, self.n_components)
        else:
            if mode == InitMode.KMeans:
                z = KMeans(self.repr_dim, self.n_components, self.metric).fit(X).assign(X)
            elif mode == InitMode.KMeans1D:
                z = KMeans1D(self.repr_dim, self.n_components, self.metric).fit(X).assign(X)
            else:
                raise NotImplementedError(f'Unknown initialization mode: {mode}')

            obs = compute_params_hard_assignment(X, z, self.n_components)

        self.update_params(obs)

    def update_params(self, obs: DPMMObs):
        pis_post = self.mhmc.pi_prior.compute_posterior(obs.Ns)
        mus_post, covs_post = self.mhmc.mu_cov_prior.compute_posterior_mv(obs.Ns, obs.mus, obs.covs)
        self._set_params(pis_post, mus_post, covs_post)

    def compute_params(self, X: Tensor, r: Tensor) -> DPMMObs:
        obs = compute_params_soft_assignment(X, r, self.n_components)
        return obs

    def estimate_assignment(self, X: Tensor) -> Tensor:
        weighted_r_E = self.estimate_log_prob(X)
        r_E = torch.softmax(weighted_r_E, dim=-1)
        return r_E

    def estimate_log_prob(self, X: Tensor) -> Tensor:
        """
        > For each data point, we calculate the log probability of that data point under each component, and then we
         weight each of those log probabilities by the probability of that topic.

        :param X: Input data points.
        :type X: Tensor
        :return: The log probability of the data given the model.
        """
        weighted_r_E = torch.stack([
            torch.log(pi) + component.log_prob(X)
            for pi, component in zip(self.pis, self.components)
        ], dim=1)
        return weighted_r_E

    def compute_loss(self, r: Tensor, r_E: Tensor) -> Tensor:
        return self.kl_div(torch.log(eps_norm(r)), eps_norm(r_E))

    @property
    def pis(self) -> Tensor:
        return self._pis.data

    @property
    def mus(self) -> Tensor:
        return self._mus.data

    @property
    def covs(self) -> Tensor:
        return self._covs.data

    def __len__(self):
        return self.n_components
