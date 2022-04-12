import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from ml.layers.dpm import initialize_kmeans, Priors, initialize_kmeans1d, initialize_soft_assignment, \
    compute_mus_soft_assignment, compute_covs_soft_assignment
from ml.layers.loss.gmm_loss import KLGMMLoss, IsoGMMLoss
from shared import get_logger

logger = get_logger(Path(__file__).stem)


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class GaussianMixtureModel(torch.nn.Module):
    pi: torch.nn.Parameter
    mus: torch.nn.Parameter
    covs: torch.nn.Parameter

    def __init__(
            self, n_components: int, repr_dim: int,
            loss: str = 'kl',
            sim='euclidean',
            mu_init=None, covs_init=None
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.repr_dim = repr_dim
        self.sim = sim

        if loss == 'kl':
            self.loss_fn = KLGMMLoss()
        elif loss == 'iso':
            self.loss_fn = IsoGMMLoss()
        else:
            raise NotImplementedError

        self._init_net(mu_init, covs_init)

    def _init_net(self, mu_init, covs_init):
        if mu_init is None:
            mu_init = torch.randn(self.n_components, self.repr_dim)
        assert mu_init.shape == (self.n_components, self.repr_dim)
        self.mus = torch.nn.Parameter(mu_init, requires_grad=False)

        if covs_init is None:
            covs_init = torch.eye(self.repr_dim).reshape(1, self.repr_dim, self.repr_dim).repeat(self.n_components, 1,
                                                                                                 1)
        assert covs_init.shape == (self.n_components, self.repr_dim, self.repr_dim)
        self.covs = torch.nn.Parameter(covs_init, requires_grad=False)

        self.pi = torch.nn.Parameter(torch.Tensor(self.n_components, 1), requires_grad=False).fill_(
            1. / self.n_components)

    def initialize_params(self, X: Tensor, r: Optional[Tensor], prior: Priors, mode='kmeans'):
        if mode == 'kmeans':
            mus, covs, pi = initialize_kmeans(X, self.n_components, prior, sim=self.sim)
        elif mode == 'kmeans1d':
            mus, covs, pi = initialize_kmeans1d(X, self.n_components, prior, sim=self.sim)
        elif mode == 'soft_assignment':
            mus, covs, pi = initialize_soft_assignment(X, r, self.n_components, prior)
        else:
            raise NotImplementedError(f'Unknown initialization mode: {mode}')

        self.mus.data, self.covs.data, self.pi.data = mus, covs, pi

    def update_params(self, X: Tensor, r: Tensor, prior: Priors):
        pi = r.sum(dim=0) / len(r)

        mus = compute_mus_soft_assignment(X, r, self.n_components)
        covs = compute_covs_soft_assignment(X, r, self.n_components, mus)

        pi = prior.compute_post_pi(pi)
        mus = prior.compute_post_mus(pi * len(r), mus)
        r_tot = r.sum(dim=0)
        covs = torch.stack([
            prior.compute_post_cov(r_tot[i], mus[i], covs[i])
            for i in range(self.n_components)
        ])

        self.mus.data, self.covs.data, self.pi.data = mus, covs, pi

    def estimate_log_prob(self, x: Tensor):
        weighted_r_E = []
        for k in range(self.n_components):
            gmm_k = MultivariateNormal(self.mus[k], self.covs[k])
            prob_k = gmm_k.log_prob(x.detach())
            weighted_r_E.append(prob_k + torch.log(self.pi[k]))

        weighted_r_E = torch.stack(weighted_r_E, dim=1)
        max_values, _ = weighted_r_E.max(dim=1, keepdim=True)
        # r_E_norm = torch.log(torch.sum(torch.exp(weighted_r_E - max_values), dim=1, keepdim=True)) + max_values
        r_E_norm = torch.logsumexp(weighted_r_E - max_values, dim=1, keepdim=True) + max_values
        r_E = torch.exp(weighted_r_E - r_E_norm)

        return r_E

    def e_step(self, X: Tensor, r: Tensor):
        loss = self.loss_fn(self, X, r)
        return loss

    def m_step(self, X: Tensor, r: Tensor, prior: Priors):
        self.update_params(X, r, prior)

    def tot_components(self):
        return self.n_components


class StackedGaussianMixtureModel(torch.nn.Module):
    def __init__(
            self, n_components: int, n_subcomponents: int, repr_dim: int,
            loss: str = 'iso',
            sim='euclidean', mu_init=None, covs_init=None
    ) -> None:
        super().__init__()
        assert loss == 'iso', 'Only isotropic GMMs are supported'

        self.repr_dim = repr_dim
        self.n_components = n_components
        self.n_subcomponents = n_subcomponents
        self.components = torch.nn.ModuleList([
            GaussianMixtureModel(
                n_subcomponents, repr_dim,
                loss='iso', sim=sim,
                mu_init=mu_init[i] if mu_init else None, covs_init=covs_init[i] if covs_init else None
            )
            for i in range(n_components)
        ])

        self.loss_fn = IsoGMMLoss(sim=sim)

    def initialize_params(self, X: Tensor, r: Optional[Tensor], prior: Priors, mode='kmeans1d'):
        z = r.argmax(dim=-1)

        n_empty = 0
        for i, submodel in enumerate(self.components):
            X_k = X[z == i]
            n_empty += (len(X_k) == 0)
            submodel.initialize_params(X_k if len(X_k) >= self.n_subcomponents else X, None, prior, mode=mode)

        if n_empty > 0:
            logging.warning(f'Encountered {n_empty} empty clusters while initializing subclusters')

    def update_params(self, X: Tensor, r: Tensor, ri: Tensor, prior: Priors):
        z = r.argmax(dim=-1)

        pi = ri.sum(dim=0) / len(ri)
        pi = prior.compute_post_pi(pi)

        for i, submodel in enumerate(self.components):
            X_k = X[z == i]
            r_k = ri[z == i, self.n_subcomponents * i: self.n_subcomponents * (i + 1)]
            z_k = r_k.argmax(dim=-1)
            denom = r_k.sum(dim=0)

            if len(X_k) < self.n_subcomponents or (denom == 0).any() or len(torch.unique(z_k)) < self.n_subcomponents:
                if len(X_k) < self.n_subcomponents:
                    logger.warning(f'Encountered empty cluster {i} while updating subclusters. Reinitializing')
                    submodel.initialize_params(X, None, prior, mode='kmeans1d')
                    submodel.pi.data = torch.tensor([0, len(X_k)]) / len(X)
                else:
                    logger.warning(f'Encountered concentrated cluster {i} while updating subclusters. Reinitializing')
                    submodel.initialize_params(X_k, None, prior, mode='kmeans1d')
            else:
                pi_k = pi[self.n_subcomponents * i: self.n_subcomponents * (i + 1)]
                mus_k = compute_mus_soft_assignment(X_k, r_k, self.n_subcomponents)
                covs_k = compute_covs_soft_assignment(X_k, r_k, self.n_subcomponents, mus_k)

                r_tot = ri.sum(dim=0)
                covs_k = torch.stack([
                    prior.compute_post_cov(r_tot[i], mus_k[i], covs_k[i])
                    for i in range(self.n_subcomponents)
                ])
                mus_k = prior.compute_post_mus(pi_k * len(ri), mus_k)

                submodel.mus.data, submodel.covs.data, submodel.pi.data = mus_k, covs_k, pi_k

    def e_step(self, X: Tensor, ri: Tensor):
        loss = self.loss_fn(self, X, ri)
        return loss

    def m_step(self, X: Tensor, r: Tensor, ri: Tensor, prior: Priors):
        self.update_params(X, r, ri, prior)

    @property
    def pi(self):
        return torch.cat([submodel.pi.data for submodel in self.components], dim=0)

    @property
    def mus(self):
        return torch.cat([submodel.mus.data for submodel in self.components], dim=0)

    @property
    def covs(self):
        return torch.cat([submodel.covs.data for submodel in self.components], dim=0)

    def component(self, i) -> GaussianMixtureModel:
        return self.components[i]

    def tot_components(self):
        return self.n_components * self.n_subcomponents
