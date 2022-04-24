from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import torch
import torchmetrics
from torch import Tensor

from ml.algo.dpm import InitMode, MHMC, DirichletPrior, NIWPrior, DPMM, StackedDPMM, BurnInMonitor, DPMMObs, \
    merge_params, DPMMParams
from ml.algo.dpm.stochastic import DPMMObsMeanFilter
from ml.utils import Metric, HParams, mask_from_idx
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class Action(IntEnum):
    Split = 0
    Merge = 1
    NoAction = 2


@dataclass
class DPMMSCModelParams(HParams):
    init_k: int = 2
    subcluster: bool = True
    mutate: bool = True
    metric: Metric = Metric.DOTP

    burnin_patience: int = 3
    early_stopping_patience: int = 5

    prior_alpha: float = 10
    prior_nu: float = 3
    prior_kappa: float = 0.0001
    prior_sigma_scale: float = 0.005

    ds_scale: float = 1.0
    min_split_points: int = 6
    n_merge_neighbors: int = 3

    cluster_init_mode: InitMode = InitMode.KMeans
    subcluster_init_mode: InitMode = InitMode.KMeans1D


class DPMMSCModel(torch.nn.Module):
    def __init__(self, repr_dim: int, hparams: DPMMSCModelParams) -> None:
        super().__init__()
        self.hparams = hparams
        self.repr_dim = repr_dim

        self.mhmc = MHMC(
            ds_scale=self.hparams.ds_scale,
            pi_prior=DirichletPrior.from_params(hparams.prior_alpha),
            mu_cov_prior=NIWPrior.from_params(
                torch.zeros(self.repr_dim), torch.eye(self.repr_dim, self.repr_dim),
                hparams.prior_nu, hparams.prior_kappa
            )
        )

        self.clusters = DPMM(hparams.init_k, self.repr_dim, self.hparams.metric, self.mhmc)
        self.cluster_mp = DPMMObsMeanFilter(hparams.init_k, self.repr_dim)

        if self.hparams.subcluster:
            self.subclusters = StackedDPMM(hparams.init_k, 2, self.repr_dim, self.hparams.metric, self.mhmc)
            self.subcluster_mp = DPMMObsMeanFilter(hparams.init_k * 2, self.repr_dim)

        self.burnin_monitor = BurnInMonitor(self.hparams.burnin_patience, threshold=0)
        self.data_ll_monitor = torchmetrics.MeanMetric()
        self.prev_action = Action.NoAction

    @property
    def k(self) -> int:
        return self.clusters.n_components

    @property
    def cluster_params(self) -> DPMMParams:
        return DPMMParams(self.clusters.pis, self.clusters.mus, self.clusters.covs)

    @property
    def subcluster_params(self) -> DPMMParams:
        return DPMMParams(self.subclusters.pis, self.subclusters.mus, self.subclusters.covs) \
            if self.hparams.subcluster else None

    @property
    def burned_in(self) -> bool:
        return self.burnin_monitor.burned_in

    @property
    def is_initialized(self):
        return self.clusters.is_initialized and (not self.hparams.subcluster or self.subclusters.is_initialized)

    @property
    def is_done(self) -> bool:
        return self.burnin_monitor.counter > self.hparams.burnin_patience + self.hparams.early_stopping_patience

    def estimate_assignment(self, X: Tensor) -> Tensor:
        return self.clusters.estimate_assignment(X)

    def reinitialize(self, X: Tensor, incremental=True, z: Optional[Tensor] = None) -> None:
        if not self.clusters.is_initialized or not incremental:
            logger.info(f'Initializing clusters with {self.hparams.cluster_init_mode}')
            self.clusters.reinitialize(X, self.hparams.cluster_init_mode, z=z)

        if self.hparams.subcluster and (not self.subclusters.is_initialized or not incremental):
            logger.info(f'Initializing subclusters with {self.hparams.subcluster_init_mode}')
            r = self.clusters.estimate_assignment(X)
            self.subclusters.reinitialize(X, r, self.hparams.subcluster_init_mode, incremental=incremental)

        self.mhmc.mu_cov_prior.update(X, self.hparams.prior_sigma_scale)

    def step_e(self, X: Tensor):
        assert self.clusters.is_initialized, "Cluster model is not initialized"
        r = self.clusters.estimate_assignment(X)
        z = r.argmax(dim=-1)
        obs_c = self.clusters.compute_params(X, r)
        self.cluster_mp.push(obs_c)

        if self.hparams.subcluster:
            assert self.subclusters.is_initialized, "Subcluster model is not initialized"
            ri = self.subclusters.estimate_assignment(X, z)
            obs_sc = self.subclusters.compute_params(X, z, ri)
            self.subcluster_mp.push(obs_sc)

        # Keep track of the data log likelihood
        ll = self.clusters.estimate_log_prob(X)
        self.data_ll_monitor.update(ll.sum(dim=-1))

    def step_m(self):
        logger.info("Updating cluster params")
        obs_c = self.cluster_mp.compute()
        self.clusters.update_params(obs_c)

        if self.hparams.subcluster:
            logger.info("Updating subcluster params")
            obs_sc = self.subcluster_mp.compute()
            if (obs_sc.Ns == 0).any():
                # TODO: handle this case or basically where the N gets bit too low
                logger.warning("Some subclusters have no samples. TODO: fix this")

            self.subclusters.update_params(obs_sc)

        # Monitor Data log likelihood. If it oscillates, then we have converged
        data_ll = self.data_ll_monitor.compute()
        burned_in = self.burnin_monitor.update(data_ll)
        if burned_in:
            logger.info("Cluster params have converged")
            if self.hparams.subcluster and self.hparams.mutate:
                self.mutate(obs_c, obs_sc)

        # Reset mean parameters aggregators
        self.data_ll_monitor.reset()
        self.cluster_mp.reset(self.clusters.n_components)
        if self.hparams.subcluster:
            self.subcluster_mp.reset(self.subclusters.n_components * self.subclusters.n_subcomponents)

    def mutate(self, obs_c: DPMMObs, obs_sc: DPMMObs):
        result = False
        if self.prev_action != Action.Split:
            self.prev_action = Action.Split
            result = self.split(obs_c, obs_sc)

        elif self.prev_action != Action.Merge:
            self.prev_action = Action.Merge
            result = self.merge(obs_c, obs_sc)

        if result:
            self.burnin_monitor.reset()

    def split(self, obs_c: DPMMObs, obs_sc: DPMMObs) -> bool:
        decisions, Hs = self.mhmc.propose_splits(obs_c, obs_sc)
        logger.info("Proposed splits: \n{}".format(
            '\n'.join(map(str, enumerate(zip(decisions.tolist(), Hs.tolist()))))
        ))

        if not decisions.any():
            return False

        # Split superclusters
        old_Ns, old_mus, old_covs = obs_c.Ns[~decisions], obs_c.mus[~decisions], obs_c.covs[~decisions]

        new_Ns, new_mus, new_covs = [old_Ns], [old_mus], [old_covs]
        for i in decisions.nonzero().flatten():
            new_Ns.append(obs_sc.Ns[i * 2: (i + 1) * 2])
            new_mus.append(obs_sc.mus[i * 2: (i + 1) * 2])
            new_covs.append(obs_sc.covs[i * 2: (i + 1) * 2])

        new_Ns, new_mus, new_covs = torch.cat(new_Ns), torch.cat(new_mus), torch.cat(new_covs)
        self.clusters.update_params(DPMMObs(new_Ns, new_mus, new_covs))

        # Split subclusters
        self.subclusters.components = torch.nn.ModuleList([
            component for not_split, component in zip(~decisions, self.subclusters.components) if not_split
        ])
        for _ in decisions.nonzero().flatten():
            self.subclusters.add_component()
            self.subclusters.add_component()

        return True

    def merge(self, obs_c: DPMMObs, _obs_sc: DPMMObs) -> bool:
        if self.k < 2:
            return False

        pairs, Hs = self.mhmc.propose_merges(obs_c)
        logger.info("Proposed merges: \n{}".format(
            '\n'.join(map(str, zip(pairs.tolist(), Hs.tolist())))
        ))

        if len(pairs) == 0:
            return False

        decisions = mask_from_idx(pairs.flatten(), self.k)

        # Merge superclusters
        old_Ns, old_mus, old_covs = obs_c.Ns[~decisions], obs_c.mus[~decisions], obs_c.covs[~decisions]
        new_Ns, new_mus, new_covs = [old_Ns], [old_mus], [old_covs]
        for pair in pairs:
            params = merge_params(obs_c.Ns[pair], obs_c.mus[pair], obs_c.covs[pair])
            new_Ns.append(params.Ns)
            new_mus.append(params.mus)
            new_covs.append(params.covs)

        new_Ns, new_mus, new_covs = torch.cat(new_Ns), torch.cat(new_mus), torch.cat(new_covs)
        self.clusters.update_params(DPMMObs(new_Ns, new_mus, new_covs))

        # Merge subclusters
        self.subclusters.components = torch.nn.ModuleList([
            component for not_merged, component in zip(~decisions, self.subclusters.components) if not_merged
        ])
        for pair in pairs:
            self.subclusters.add_component().update_params(DPMMObs(
                obs_c.Ns[pair], obs_c.mus[pair], obs_c.covs[pair]
            ))

        return True
