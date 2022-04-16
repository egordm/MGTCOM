from pathlib import Path
from typing import Union, List

import torch
from torch import Tensor
from torch.nn import ModuleList

from ml.algo.dpm.dpmm import DirichletProcessMixtureModel, InitMode
from ml.algo.dpm.mhmc import MHMC
from ml.algo.dpm.statistics import DPMMObs
from ml.utils import Metric, unique_count
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class StackedDirichletProcessMixtureModel(torch.nn.Module):
    components: Union[List[DirichletProcessMixtureModel], ModuleList]

    def __init__(self, n_components: int, n_subcomponents: int, repr_dim: int, metric: Metric, mhmc: MHMC):
        super().__init__()
        self.n_components = n_components
        self.n_subcomponents = n_subcomponents
        self.repr_dim = repr_dim
        self.metric = metric
        self.mhmc = mhmc

        self.components = torch.nn.ModuleList([
            DirichletProcessMixtureModel(n_subcomponents, repr_dim, self.metric, self.mhmc)
            for _ in range(n_components)
        ])

    @property
    def is_initialized(self) -> bool:
        return all([c.is_initialized for c in self.components])

    def reinitialize(self, X: Tensor, r: Tensor, mode: InitMode = InitMode.KMeans, incremental=False):
        z = r.argmax(dim=1)
        for i, component in enumerate(self.components):
            X_k = X[z == i]
            r_k = r[z == i]
            if len(X_k) < self.n_subcomponents:
                logger.warning(f'Encountered empty cluster {i} while updating subclusters. Reinitializing')
                component.reinitialize(X, r_k, mode=mode)
            elif not self.is_initialized or not incremental:
                component.reinitialize(X_k, r_k, mode=mode)

    def update_params(self, obs: DPMMObs):
        for i, component in enumerate(self.components):
            component.update_params(
                DPMMObs(
                    obs.Ns[i * self.n_subcomponents:(i + 1) * self.n_subcomponents],
                    obs.mus[i * self.n_subcomponents:(i+1) * self.n_subcomponents, :],
                    obs.covs[i * self.n_subcomponents:(i+1) * self.n_subcomponents, :, :],
                ),
            )

    def compute_params(self, X: Tensor, z: Tensor, r: Tensor) -> DPMMObs:
        Ns = unique_count(z, self.n_components)

        Ns_K = torch.zeros(self.n_components * self.n_subcomponents)
        mus_K = torch.zeros(self.n_components * self.n_subcomponents, self.repr_dim)
        covs_K = torch.zeros(self.n_components * self.n_subcomponents, self.repr_dim, self.repr_dim)

        for i, (component, N_k) in enumerate(zip(self.components, Ns)):
            X_k, r_k = X[z == i], r[z == i]
            (
                Ns_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents],
                mus_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents, :],
                covs_K[i * self.n_subcomponents: (i + 1) * self.n_subcomponents, :, :]
            ) = component.compute_params(X_k, r_k)

        return DPMMObs(Ns_K, mus_K, covs_K)

    def estimate_log_prob(self, X: Tensor, z: Tensor) -> Tensor:
        Ns = unique_count(z, self.n_components)
        r_E = torch.zeros(X.shape[0], self.n_subcomponents)
        for i, (component, N_k) in enumerate(zip(self.components, Ns)):
            if N_k > 0:
                r_E[z == i] = component.estimate_assignment(X[z == i])

        return r_E

    def add_component(self) -> DirichletProcessMixtureModel:
        component = DirichletProcessMixtureModel(
            self.n_subcomponents, self.repr_dim, self.metric, self.mhmc
        )
        self.components.append(component)
        self.n_components = len(self.components)
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

    def __getitem__(self, item) -> DirichletProcessMixtureModel:
        return self.components[item]

    def __len__(self):
        return self.n_components

