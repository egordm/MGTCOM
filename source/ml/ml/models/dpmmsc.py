from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Any

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor

from ml.algo.dpm import BurnInMonitor, DPMMObs, merge_params, StackedDPMM, \
    DPMM, NIWPrior, DirichletPrior, MHMC, InitMode, DPMMParams
from ml.algo.dpm.stochastic import DPMMObsMeanFilter
from ml.utils import HParams, Metric, mask_from_idx
from ml.utils.outputs import OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class Action(IntEnum):
    Split = 0
    Merge = 1
    NoAction = 2


class Stage(IntEnum):
    GatherSamples = 0
    BurnIn = 1
    Mutation = 2


@dataclass
class DPMMSCModelParams(HParams):
    init_k: int = 2
    subcluster: bool = True
    mutate: bool = True
    metric: Metric = Metric.L2

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


class DPMMSubClusteringModel(pl.LightningModule):
    hparams: DPMMSCModelParams
    train_samples: List[Tensor]

    val_outputs: Dict[str, Tensor] = None
    test_outputs: Dict[str, Tensor] = None
    pred_outputs: Dict[str, Tensor] = None

    def __init__(self, repr_dim: int, hparams: DPMMSCModelParams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())

        self.repr_dim = repr_dim
        self.mhmc = MHMC(
            ds_scale=self.hparams.ds_scale,
            pi_prior=DirichletPrior.from_params(hparams.prior_alpha),
            mu_cov_prior=NIWPrior.from_params(
                torch.zeros(self.repr_dim), torch.eye(self.repr_dim, self.repr_dim),
                hparams.prior_nu, hparams.prior_kappa
            )
        )

        self.clusters = DPMM(
            hparams.init_k, self.repr_dim, self.hparams.metric, self.mhmc
        )
        self.cluster_mp = DPMMObsMeanFilter(hparams.init_k, self.repr_dim)

        if self.hparams.subcluster:
            self.subclusters = StackedDPMM(
                hparams.init_k, 2, self.repr_dim, self.hparams.metric, self.mhmc
            )
            self.subcluster_mp = DPMMObsMeanFilter(hparams.init_k * 2, self.repr_dim)

        self.burnin_monitor = BurnInMonitor(self.hparams.burnin_patience, threshold=0)
        self.data_ll = torchmetrics.MeanMetric()

        self.stage = Stage.GatherSamples
        self.prev_action = Action.NoAction
        self.train_samples = []

    @property
    def k(self):
        return self.clusters.n_components

    def forward(self, X):
        out = {
            'X': X,
        }

        if self.clusters.is_initialized:
            r = self.clusters.estimate_assignment(X)
            z = r.argmax(dim=-1)
            out.update(dict(r=r.detach(), z=z.detach()))

            if self.hparams.subcluster and self.subclusters.is_initialized:
                ri = self.subclusters.estimate_assignment(X, z)
                zi = ri.argmax(dim=-1)
                out.update(dict(ri=ri.detach(), zi=zi.detach()))

        return out

    def on_train_epoch_start(self) -> None:
        if self.stage == Stage.GatherSamples and len(self.train_samples) > 0:
            X = torch.cat(self.train_samples, dim=0)

            if not self.clusters.is_initialized:
                logger.info(f'Initializing clusters with {self.hparams.cluster_init_mode}')
                self.clusters.reinitialize(X, None, self.hparams.cluster_init_mode)

            if self.hparams.subcluster and not self.subclusters.is_initialized:
                logger.info(f'Initializing subclusters with {self.hparams.subcluster_init_mode}')
                r = self.clusters.estimate_assignment(X)
                self.subclusters.reinitialize(X, r, self.hparams.subcluster_init_mode, incremental=True)

            self.mhmc.mu_cov_prior.update(X, self.hparams.prior_sigma_scale)
            self.stage = Stage.BurnIn

        self.train_samples = []
        self.cluster_mp.reset(self.clusters.n_components)
        if self.hparams.subcluster:
            self.subcluster_mp.reset(self.subclusters.n_components * self.subclusters.n_subcomponents)

        self.log_dict({
            'stage_g': self.stage == Stage.GatherSamples,
            'stage_b': self.stage == Stage.BurnIn,
            'stage_m': self.stage == Stage.Mutation,
            'k': self.k,
            'bi_count': self.burnin_monitor.counter,
        }, prog_bar=True)

    def on_train_batch_start(self, batch, batch_idx: int, unused: int = 0):
        if self.burnin_monitor.counter > self.hparams.early_stopping_patience + self.hparams.burnin_patience:
            return -1

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        X = batch.detach()

        if self.stage == Stage.GatherSamples:
            self.train_samples.append(X.cpu())

        if self.clusters.is_initialized:
            r = self.clusters.estimate_assignment(X)
            z = r.argmax(dim=-1)
            obs_c = self.clusters.compute_params(X, r.detach())
            self.cluster_mp.push(obs_c)

            if self.hparams.subcluster and self.subclusters.is_initialized:
                ri = self.subclusters.estimate_assignment(X, z.detach())
                obs_sc = self.subclusters.compute_params(X, z.detach(), ri.detach())
                self.subcluster_mp.push(obs_sc)

            ll = self.clusters.estimate_log_prob(X)
            self.data_ll.update(ll.sum(dim=-1))

    def training_epoch_end(self, outputs):
        obs_c, obs_sc, burned_in = None, None, False

        if self.stage >= Stage.BurnIn:
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
            data_ll = self.data_ll.compute()
            self.log('data_ll', self.data_ll, prog_bar=True)
            burned_in = self.burnin_monitor.update(data_ll)
            if burned_in:
                logger.info("Cluster params have converged")
                self.stage = Stage.Mutation

        if self.hparams.subcluster and self.hparams.mutate and self.stage == Stage.Mutation:
            # TODO: to do splits
            if self.prev_action != Action.Split:
                self.prev_action = Action.Split
                self.split(obs_c, obs_sc)

            elif self.prev_action != Action.Merge:
                self.prev_action = Action.Merge
                self.merge(obs_c, obs_sc)

    def split(self, obs_c: DPMMObs, obs_sc: DPMMObs):
        decisions = self.mhmc.propose_splits(obs_c, obs_sc)
        if not decisions.any():
            return

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

        self.stage = Stage.GatherSamples
        self.burnin_monitor.reset()

    def merge(self, obs_c: DPMMObs, _obs_sc: DPMMObs):
        if self.k < 2:
            return

        pairs = self.mhmc.propose_merges(obs_c)
        if len(pairs) == 0:
            return

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

        self.stage = Stage.BurnIn
        self.burnin_monitor.reset()

    def _eval_epoch_end(self, outputs):
        outputs = OutputExtractor(outputs)
        out = dict(
            X=outputs.extract_cat('X'),
        )

        if not self.stage == Stage.GatherSamples:
            r = outputs.extract_cat('r').cpu()
            out.update(dict(r=r))

            if self.hparams.subcluster:
                ri = outputs.extract_cat('ri').cpu()
                out.update(dict(ri=ri))

        return out

    @property
    def is_subclustering(self):
        return self.hparams.subcluster and self.subclusters.is_initialized

    @property
    def cluster_params(self) -> DPMMParams:
        return DPMMParams(self.clusters.pis, self.clusters.mus, self.clusters.covs)

    @property
    def subcluster_params(self) -> DPMMParams:
        return DPMMParams(self.subclusters.pis, self.subclusters.mus, self.subclusters.covs) \
            if self.is_subclustering else None

    @property
    def samplespace_changed(self) -> bool:
        return False

    def estimate_assignment(self, X: torch.Tensor) -> torch.Tensor:
        return self.clusters.estimate_assignment(X)

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_epoch_end(self, outputs):
        self.val_outputs = self._eval_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.forward(batch)

    def test_epoch_end(self, outputs):
        self.test_outputs = self._eval_epoch_end(outputs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
