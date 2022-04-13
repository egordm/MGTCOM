from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Optional, List

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from ml.layers.clustering import KMeans, KMeans1D
from ml.layers.dpm import Priors
from ml.layers.loss.gmm_loss import KLGMMLoss, IsoGMMLoss
from ml.utils import unique_count, compute_cov
from shared import get_logger

logger = get_logger(Path(__file__).stem)

GMMParams = namedtuple('GMMParams', ['pi', 'mus', 'covs'])


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class InitMode(Enum):
    KMeans = 'kmeans'
    KMeans1D = 'kmeans1d'
    SoftAssignment = 'soft_assignment'


class GMMLoss(Enum):
    KL = 'kl'
    Iso = 'iso'


class GaussianMixtureModel(torch.nn.Module):
    components: List[MultivariateNormal]

    def __init__(
            self, n_components: int, repr_dim: int,
            loss: str = 'kl', sim='euclidean', init_mode: InitMode = InitMode.KMeans,
            init_params: GMMParams = None,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.repr_dim = repr_dim
        self.sim = sim
        self.init_mode = init_mode

        if loss == 'kl':
            self.loss_fn = KLGMMLoss()
        elif loss == 'iso':
            self.loss_fn = IsoGMMLoss()
        else:
            raise NotImplementedError

        self._pi = torch.nn.Parameter(torch.ones(self.n_components) / self.n_components, requires_grad=False)
        self._mus = torch.nn.Parameter(torch.randn(self.n_components, self.repr_dim), requires_grad=False)
        self._covs = torch.nn.Parameter(
            torch.eye(self.repr_dim).reshape(1, self.repr_dim, self.repr_dim).repeat(self.n_components, 1, 1),
            requires_grad=False)

        if init_params is not None:
            self.set_params(*init_params)

    def set_params(self, pi: Tensor, mus: Tensor, covs: Tensor):
        self.n_components = len(pi)
        assert pi.shape == (self.n_components,)
        assert mus.shape == (self.n_components, self.repr_dim)
        assert covs.shape == (self.n_components, self.repr_dim, self.repr_dim)

        self._pi.data = pi
        self._mus.data = mus
        self._covs.data = covs
        self.components = [MultivariateNormal(mu, cov) for mu, cov in zip(mus, covs)]

        return self

    def reinit_params(self, X: Tensor, prior: Priors):
        if self.init_mode == InitMode.KMeans:
            z = KMeans(self.repr_dim, self.n_components, sim=self.sim).fit(X).assign(X)
        elif self.init_mode == InitMode.KMeans1D:
            z = KMeans1D(self.repr_dim, self.n_components, sim=self.sim).fit(X).assign(X)
        else:
            raise NotImplementedError(f'Unknown initialization mode: {self.init_mode}')

        (_, D) = X.shape
        pi, mus, covs = compute_hard_assignment(X, z, self.n_components, prior.mus_covs_prior.psi)
        pi_, mus, covs = prior.compute_post_params(D, pi * len(X), pi, mus, covs)
        self.set_params(pi, mus, covs)

    def update_params(self, X: Tensor, r: Tensor, prior: Priors):
        (_, D) = X.shape
        pi, mus, covs = compute_soft_assignment(X, r, self.n_components)
        pi, mus, covs = prior.compute_post_params(D, r.sum(dim=0), pi, mus, covs)
        self.set_params(pi, mus, covs)

    def estimate_log_prob(self, x: Tensor):
        weighted_r_E = torch.stack([
            torch.log(pi_k) + component.log_prob(x.detach())
            for pi_k, component in zip(self.pi, self.components)
        ], dim=1)

        max_values, _ = weighted_r_E.max(dim=1, keepdim=True)
        r_E_norm = torch.logsumexp(weighted_r_E - max_values, dim=1, keepdim=True) + max_values
        r_E = torch.exp(weighted_r_E - r_E_norm)

        return r_E

    def e_step(self, X: Tensor, r: Tensor):
        loss = self.loss_fn(self, X, r)
        return loss

    def m_step(self, X: Tensor, r: Tensor, prior: Priors):
        self.update_params(X, r, prior)

    def __len__(self):
        return self.n_components

    @property
    def pi(self):
        return self._pi.data

    @property
    def mus(self):
        return self._mus.data

    @property
    def covs(self):
        return self._covs.data


class StackedGaussianMixtureModel(torch.nn.Module):
    def __init__(
            self, n_components: int, n_subcomponents: int, repr_dim: int,
            loss: str = 'iso', sim='euclidean', init_mode: InitMode = InitMode.KMeans1D,
            init_params: List[GMMParams] = None
    ) -> None:
        super().__init__()

        self.repr_dim = repr_dim
        self.n_subcomponents = n_subcomponents
        self.sim = sim
        self.loss = loss
        self.init_mode = init_mode

        assert loss == 'iso', 'Only isotropic GMMs are supported'
        self.loss_fn = IsoGMMLoss(sim=sim)

        self.components = torch.nn.ModuleList([
            GaussianMixtureModel(
                n_subcomponents, repr_dim,
                loss=self.loss, sim=sim, init_mode=self.init_mode,
                init_params=init_params[i] if init_params is not None else None
            )
            for i in range(n_components)
        ])

    def reinit_params(self, X: Tensor, r: Optional[Tensor], prior: Priors):
        z = r.argmax(dim=-1)
        N_K = unique_count(z, len(self))

        for i, (N_k, component) in enumerate(zip(N_K, self.components)):
            if N_k < self.n_subcomponents:
                logger.warning(f'Encountered empty cluster {i} while updating subclusters. Reinitializing')

            component.reinit_params(X[z == i] if N_k >= self.n_subcomponents else X, prior)

    def update_params(self, X: Tensor, r: Tensor, ri: Tensor, prior: Priors):
        z = r.argmax(dim=-1)

        for i, submodel in enumerate(self.components):
            X_k = X[z == i]
            r_k = ri[z == i, self.n_subcomponents * i: self.n_subcomponents * (i + 1)]
            z_k = r_k.argmax(dim=-1)
            N_k = r_k.sum(dim=0)

            if len(X_k) < self.n_subcomponents or (N_k == 0).any() or len(torch.unique(z_k)) < self.n_subcomponents:
                logger.warning('Encountered {} cluster {} while updating subclusters. Reinitializing'
                               .format('empty' if len(X_k) < self.n_subcomponents else 'concentrated', i))
                submodel.reinit_params(X_k if len(X_k) >= self.n_subcomponents else X, prior)
            else:
                submodel.update_params(X_k, r_k, prior)

    def add_component(self, init_params: GMMParams = None) -> GaussianMixtureModel:
        component = GaussianMixtureModel(
            self.n_subcomponents, self.repr_dim,
            loss='iso', sim=self.sim,
            init_params=init_params
        )
        self.components.append(component)
        return component

    def e_step(self, X: Tensor, ri: Tensor):
        loss = self.loss_fn(self, X, ri)
        return loss

    def m_step(self, X: Tensor, r: Tensor, ri: Tensor, prior: Priors):
        self.update_params(X, r, ri, prior)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    @property
    def n_components(self):
        return len(self.components) * self.n_subcomponents

    @property
    def pi(self):
        return torch.cat([submodel.pi for submodel in self.components], dim=0)

    @property
    def mus(self):
        return torch.cat([submodel.mus for submodel in self.components], dim=0)

    @property
    def covs(self):
        return torch.cat([submodel.covs for submodel in self.components], dim=0)


EPS = 0.0001


def compute_hard_assignment(X: Tensor, z: Tensor, k: int, cov_init: Tensor):
    N_K = unique_count(z, k)

    pi = (N_K + EPS) / N_K.sum()
    mus = torch.zeros(k, X.shape[1]).index_add_(0, z, X) / N_K.unsqueeze(1)
    covs = torch.stack([
        compute_cov(X[z == i] - mus[i].unsqueeze(0)) if N_K[i] > 0 else cov_init
        for i in range(k)
    ])
    return pi, mus, covs


def compute_soft_assignment(X: Tensor, r: Tensor, k: int):
    N_K = r.sum(dim=0) + EPS

    pi = N_K / len(r)
    mus = torch.stack([
        (r[:, i].unsqueeze(1) * X).sum(dim=0) / N_K[i]
        for i in range(k)
    ])
    covs = compute_covs_soft_assignment(N_K, X, r, k, mus)
    return pi, mus, covs


def compute_covs_soft_assignment(N_K, X: Tensor, r: Tensor, k: int, mus: Tensor):
    covs = []
    for i in range(k):
        X_diff = X - mus[i].unsqueeze(0)
        if len(X) > 0:
            covs.append(torch.matmul((r[:, i].unsqueeze(0) * X_diff.T), X_diff) / N_K[i])
        else:
            covs.append(torch.eye(mus.shape[1]) * EPS)
    return torch.stack(covs)
