from pathlib import Path
from typing import Union, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList

from ml.algo.dpm.dpmm import DirichletProcessMixtureModel, InitMode
from ml.algo.dpm.priors import DirichletPrior, NIWPrior, MultivarNormalParams
from ml.layers.dpm import compute_params_soft_assignment
from ml.utils import Metric, unique_count
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class StackedDirichletProcessMixtureModel(torch.nn.Module):
    components: Union[List[DirichletProcessMixtureModel], ModuleList]

    def __init__(
            self,
            n_components: int, n_subcomponents: int, repr_dim: int, metric: Metric,
            pi_prior: DirichletPrior, mu_cov_prior: NIWPrior
    ):
        super().__init__()
        self.n_components = n_components
        self.n_subcomponents = n_subcomponents
        self.repr_dim = repr_dim
        self.metric = metric
        self.pi_prior = pi_prior
        self.mu_cov_prior = mu_cov_prior

        self.components = torch.nn.ModuleList([
            DirichletProcessMixtureModel(n_subcomponents, repr_dim, self.metric, pi_prior, mu_cov_prior)
            for _ in range(n_components)
        ])
        self.is_initialized = False

    def reinitialize(self, X: Tensor, r: Tensor, mode: InitMode = InitMode.KMeans):
        z = r.argmax(dim=1)
        for i, component in enumerate(self.components):
            X_k = X[z == i]
            r_k = r[z == i]
            if len(X_k) < self.n_subcomponents:
                logger.warning(f'Encountered empty cluster {i} while updating subclusters. Reinitializing')

            component.reinitialize(X_k if len(X_k) >= self.n_subcomponents else X, r_k, mode=mode)

        self.is_initialized = True

    def update_params(self, params: MultivarNormalParams, Ns: Tensor):
        for i, component in enumerate(self.components):
            component.update_params(
                MultivarNormalParams(
                    params.mus[i * self.n_subcomponents:(i+1) * self.n_subcomponents, :],
                    params.covs[i * self.n_subcomponents:(i+1) * self.n_subcomponents, :, :],
                ),
                Ns[i * self.n_subcomponents:(i+1) * self.n_subcomponents]
            )

    def compute_params(self, X: Tensor, z: Tensor, r: Tensor) -> Tuple[Tensor, MultivarNormalParams]:
        Ns = unique_count(z, self.n_components)

        Ns_K = torch.zeros(self.n_components * self.n_subcomponents)
        mus_K = torch.zeros(self.n_components * self.n_subcomponents, self.repr_dim)
        covs_K = torch.zeros(self.n_components * self.n_subcomponents, self.repr_dim, self.repr_dim)

        for i, (component, N_k) in enumerate(zip(self.components, Ns)):
            X_k, r_k = X[z == i], r[z == i]
            (
                Ns_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents],
                (
                    mus_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents, :],
                    covs_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents, :, :]
                )
            ) = component.compute_params(X_k, r_k)

        return Ns_K, MultivarNormalParams(mus_K, covs_K)

    def estimate_log_prob(self, X: Tensor, z: Tensor) -> Tensor:
        Ns = unique_count(z, self.n_components)
        r_E = torch.zeros(X.shape[0], self.n_subcomponents)
        for i, (component, N_k) in enumerate(zip(self.components, Ns)):
            if N_k > 0:
                r_E[z == i] = component.estimate_log_prob(X[z == i])

        return r_E

    def add_component(self) -> DirichletProcessMixtureModel:
        component = DirichletProcessMixtureModel(
            self.n_subcomponents, self.n_feat, self.metric, self.pi_prior, self.mu_cov_prior
        )
        self.components.append(component)
        return component

    @property
    def pis(self) -> Tensor:
        return torch.cat([component.pis for component in self.components], dim=0)

    @property
    def mus(self) -> Tensor:
        return torch.cat([component.mus for component in self.components], dim=0)

    @property
    def covs(self) -> Tensor:
        return torch.cat([component.covs for component in self.components], dim=0)

