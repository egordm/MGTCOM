from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torchmetrics
from tch_geometric.loader import CustomLoader
from torch import Tensor
from torch.utils.data import Dataset

from ml.algo.dpm.dpmm import DirichletProcessMixtureModel, InitMode
from ml.algo.dpm.dpmm_stacked import StackedDirichletProcessMixtureModel
from ml.algo.dpm.mhmc import MHMC
from ml.algo.dpm.priors import DirichletPrior, NIWPrior
from ml.algo.dpm.statistics import merge_params, DPMMObs
from ml.algo.dpm.stochastic import DPMMObsMeanFilter
from ml.layers.dpm.burnin_monitor import BurnInMonitor
from ml.utils import HParams, Metric, OutputExtractor, DataLoaderParams, mask_from_idx
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
    init_k: int = 1
    subcluster: bool = True
    # subcluster: bool = False
    metric: Metric = Metric.L2

    burnin_patience: int = 3
    early_stopping_patience: int = 5

    prior_alpha: float = 10
    prior_nu: float = 10
    prior_kappa: float = 0.0001
    prior_sigma_scale: float = 0.005

    min_split_points: int = 6
    n_merge_neighbors: int = 3

    cluster_init_mode: InitMode = InitMode.KMeans
    subcluster_init_mode: InitMode = InitMode.KMeans1D

    loader_args: DataLoaderParams = DataLoaderParams()


class DPMMSubClusteringModel(pl.LightningModule):
    hparams: DPMMSCModelParams
    train_samples: List[Tensor]
    validation_outputs: Dict[str, Tensor]

    def __init__(self, dataset: Dataset, hparams: DPMMSCModelParams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.dataset = dataset

        self.repr_dim = dataset[[0]].shape[-1]
        self.mhmc = MHMC(
            ds_scale=1.0,
            pi_prior=DirichletPrior.from_params(hparams.prior_alpha),
            mu_cov_prior=NIWPrior.from_params(
                torch.zeros(self.repr_dim), torch.eye(self.repr_dim, self.repr_dim),
                hparams.prior_nu, hparams.prior_kappa
            )
        )

        self.clusters = DirichletProcessMixtureModel(
            hparams.init_k, self.repr_dim, self.hparams.metric, self.mhmc
        )
        self.cluster_mp = DPMMObsMeanFilter(hparams.init_k, self.repr_dim)

        if self.hparams.subcluster:
            self.subclusters = StackedDirichletProcessMixtureModel(
                hparams.init_k, 2, self.repr_dim, self.hparams.metric, self.mhmc
            )
            self.subcluster_mp = DPMMObsMeanFilter(hparams.init_k * 2, self.repr_dim)

        self.burnin_monitor = BurnInMonitor(self.hparams.burnin_patience, threshold=0)
        self.data_ll = torchmetrics.MeanMetric()

        self.stage = Stage.GatherSamples
        self.prev_action = Action.NoAction
        self.train_samples = []
        self.prev_params = None

    @property
    def k(self):
        return self.clusters.n_components

    def forward(self, X):
        out = {}

        if self.clusters.is_initialized:
            r = self.clusters.estimate_assignment(X)
            z = r.argmax(dim=-1)
            out.update(dict(r=r.detach(), z=z.detach()))

            if self.hparams.subcluster and self.subclusters.is_initialized:
                ri = self.subclusters.estimate_log_prob(X, z)
                zi = ri.argmax(dim=-1)
                out.update(dict(ri=ri.detach(), zi=zi.detach()))

        return out

    def on_train_epoch_start(self) -> None:
        self.prev_params = (self.clusters.mus.clone(), self.clusters.covs.clone())

        if self.stage == Stage.GatherSamples and len(self.train_samples) > 0:
            X = torch.cat(self.train_samples, dim=0)

            if not self.clusters.is_initialized:
                self.clusters.reinitialize(X, None, self.hparams.cluster_init_mode)

            if not self.subclusters.is_initialized:
                r = self.clusters.estimate_assignment(X)
                self.subclusters.reinitialize(X, r, self.hparams.subcluster_init_mode, incremental=True)

            self.mhmc.mu_cov_prior.update(X, self.hparams.prior_sigma_scale)
            self.stage = Stage.BurnIn

        self.train_samples = []
        self.cluster_mp.reset(self.clusters.n_components)
        if self.hparams.subcluster:
            self.subcluster_mp.reset(self.subclusters.n_components * self.subclusters.n_subcomponents)

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
            obs = self.clusters.compute_params(X, r.detach())
            self.cluster_mp.push(obs)

            ll = self.clusters.estimate_log_prob(X)
            self.data_ll.update(ll.sum(dim=-1))

            if self.hparams.subcluster and self.subclusters.is_initialized:
                ri = self.subclusters.estimate_log_prob(X, z.detach())
                obs = self.subclusters.compute_params(X, z.detach(), ri.detach())
                self.subcluster_mp.push(obs)

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
                    u = 0

                self.subclusters.update_params(obs_sc)

            # Monitor Data log likelihood. If it oscillates, then we have converged
            data_ll = self.data_ll.compute()
            self.log('data_ll', self.data_ll, prog_bar=True)
            burned_in = self.burnin_monitor.update(data_ll)
            if burned_in:
                logger.info("Cluster params have converged")
                self.stage = Stage.Mutation

        if self.stage == Stage.Mutation:
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

    def merge(self, obs_c: DPMMObs, obs_sc: DPMMObs):
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch.detach()
        out = self.forward(X)
        return out

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        outputs = OutputExtractor(outputs)
        self.validation_outputs = {}

        if not self.stage == Stage.GatherSamples:
            r = outputs.extract_cat('r').cpu()
            self.validation_outputs.update(dict(r=r))

            if self.hparams.subcluster:
                ri = outputs.extract_cat('ri').cpu()
                self.validation_outputs.update(dict(ri=ri))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return CustomLoader(self.dataset, shuffle=True, **self.hparams.loader_args)

    def val_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def predict_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)
