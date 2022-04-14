import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from simple_parsing import choice, field
from tch_geometric.loader import CustomLoader
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset

from ml.layers.dpm import ClusteringNet, SubClusteringNet, Priors, GaussianMixtureModel, StackedGaussianMixtureModel, \
    WeightsInitMode, InitMode, PriorParams
from ml.layers.dpm.burnin_monitor import BurnInMonitor
from ml.layers.dpm.mhsc_rules import MHSCRules
from ml.utils import dicts_extract, OutputExtractor, Metric
from ml.utils.config import HParams, DataLoaderParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class DPMClusteringModelParams(HParams):
    lat_dim: int = 32
    init_k: int = 1
    subcluster: bool = False

    metric: Metric = Metric.L2

    cluster_burnin_patience: int = 1
    subcluster_burnin_patience: int = 1

    prior_params: PriorParams = PriorParams()

    mu_init_fn: InitMode = InitMode.KMeans
    mu_sub_init_fn: InitMode = InitMode.KMeans1D
    mu_update_fn: InitMode = InitMode.SoftAssignment

    alpha: float = 10.0
    split_prob: Optional[float] = None
    merge_prob: Optional[float] = None
    min_split_points: int = 6
    n_merge_neighbors: int = 3
    init_weights: WeightsInitMode = WeightsInitMode.Random
    init_weights_split_sub: WeightsInitMode = WeightsInitMode.Random
    init_weights_merge_sub: WeightsInitMode = WeightsInitMode.Random

    cluster_lr: float = 0.05
    subcluster_lr: float = 0.01

    loader_args: DataLoaderParams = DataLoaderParams()


class OptimizerIdx(IntEnum):
    Cluster = 0
    SubCluster = 1


class Stage(IntEnum):
    Clustering = 1
    SubClustering = 2


class Action(IntEnum):
    Split = 0
    Merge = 1
    NoAction = 2


class DPMClusteringModel(pl.LightningModule):
    hparams: DPMClusteringModelParams

    stage: Stage
    last_action: Action
    action: Action

    val_r: Tensor
    val_ri: Tensor

    def __init__(
            self,
            dataset: Dataset, hparams: DPMClusteringModelParams,
            repr_dim: int
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.dataset = dataset

        self.k = hparams.init_k
        self.repr_dim = repr_dim

        self.cluster_net = ClusteringNet(self.k, self.repr_dim, self.hparams.lat_dim)
        self.subcluster_net = SubClusteringNet(self.k, self.repr_dim, self.hparams.lat_dim) if self.hparams.subcluster else None

        self.prior = Priors(repr_dim, PriorParams(**self.hparams.prior_params))
        self.cluster_gmm = GaussianMixtureModel(
            self.k, self.repr_dim, metric=self.hparams.metric, loss='kl', init_mode=self.hparams.mu_init_fn
        )
        self.subcluster_gmm = StackedGaussianMixtureModel(
            self.k, 2, self.repr_dim, metric=self.hparams.metric, loss='iso', init_mode=self.hparams.mu_sub_init_fn
        ) if self.hparams.subcluster else None

        self.ms_rules = MHSCRules(
            self.prior, self.cluster_gmm, self.subcluster_gmm,
            self.hparams.alpha, self.hparams.split_prob, self.hparams.merge_prob,
            self.hparams.min_split_points, self.hparams.n_merge_neighbors,
            metric=self.hparams.metric,
        )

        self.stage = Stage.Clustering
        # self.last_action = Action.NoAction
        self.last_action = Action.Split

        self.cluster_burnin = BurnInMonitor(self.hparams.cluster_burnin_patience)
        self.cluster_m_burnin = BurnInMonitor(self.hparams.cluster_burnin_patience)
        self.subcluster_burnin = BurnInMonitor(self.hparams.subcluster_burnin_patience)
        self.subcluster_m_burnin = BurnInMonitor(self.hparams.subcluster_burnin_patience)


    def forward(self, batch: Tensor):
        return self.cluster_net(batch)

    def on_train_start(self):
        # Initialize GMM on the data
        xs = self.dataset[torch.arange(len(self.dataset), dtype=torch.long)]
        self.prior.init_priors(xs)

        logger.info(f"Initializing cluster params")
        self.cluster_gmm.reinit_params(xs, self.prior)
        self.cluster_burnin.reset()

    def on_train_epoch_start(self) -> None:
        if self.hparams.subcluster and self.stage == Stage.Clustering and self.cluster_burnin.burned_in:
            self.stage = Stage.SubClustering

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        out = {}
        x = batch
        r = self.cluster_net(x)

        if optimizer_idx == OptimizerIdx.Cluster:
            cluster_loss, cluster_cl = self.cluster_gmm.e_step(x, r)
            loss = cluster_loss

            out.update(dict(
                r=r.detach(),
                x=x.detach(),
                cluster_loss=cluster_loss.detach(),
                # cluster_cl=cluster_cl.detach(),
            ))
        elif optimizer_idx == OptimizerIdx.SubCluster and self.stage == Stage.SubClustering:
            z = r.detach().argmax(dim=-1)
            ri = self.subcluster_net(x, z)

            subcluster_loss, subcluster_cl = self.subcluster_gmm.e_step(x, z, ri)
            loss = subcluster_loss

            out.update(dict(
                ri=ri.detach(),
                subcluster_loss=subcluster_loss.detach(),
                # subcluster_cl=subcluster_cl.detach(),
            ))
        else:
            return None

        out['loss'] = loss
        return out

    def training_epoch_end(self, outputs) -> None:
        outputs = OutputExtractor(outputs)

        X, r = outputs.extract_cat('x'), outputs.extract_cat('r')

        # Cluster Monitoring
        cluster_loss, cluster_cl = outputs.extract_mean('cluster_loss'), 0 #, outputs.extract_mean('cluster_cl')
        cluster_burned_in, subcluster_burned_in = self.cluster_burnin.update(cluster_loss), False
        cluster_m_burned_in, subcluster_m_burned_in = cluster_burned_in and self.cluster_m_burnin.update(cluster_loss), False

        # Subcluster Monitoring
        if self.stage == Stage.SubClustering:
            subcluster_loss, subcluster_cl = outputs.extract_mean('subcluster_loss'), 0 #, outputs.extract_mean('subcluster_cl')
            subcluster_burned_in = cluster_m_burned_in and self.subcluster_burnin.update(subcluster_loss)
            subcluster_m_burned_in = subcluster_burned_in and self.subcluster_m_burnin.update(subcluster_loss)
            ri = outputs.extract_cat('ri')
        else:
            subcluster_loss, subcluster_cl = float('nan'), float('nan')
            ri = None

        # Decide on a Merge or Split action
        action = Action.NoAction
        if subcluster_m_burned_in and self.stage >= Stage.SubClustering:
            if self.last_action != Action.Split:
                action = Action.Split
                self.last_action = Action.Split
            elif self.last_action != Action.Merge:
                action = Action.Merge
                self.last_action = Action.Merge

        # Clustering M step
        if cluster_burned_in:
            logger.info("Updating cluster params")
            self.cluster_gmm.m_step(X, r, self.prior)

        # Subcluster M step
        # if self.stage == Stage.SubClustering and subcluster_burned_in:
        if self.stage == Stage.SubClustering and subcluster_burned_in:
            logger.info("Updating subcluster params")
            self.subcluster_gmm.m_step(X, r, ri, self.prior)

        # Initialize subcluster GMM (once clusters have burned in)
        if self.hparams.subcluster and self.stage == Stage.Clustering and cluster_burned_in:
            logger.info("Initializing subcluster params")
            self.stage = Stage.SubClustering
            self.subcluster_gmm.reinit_params(X, r, self.prior)

        if action == Action.Split:
            self.split(X, r, ri)
        elif action == Action.Merge:
            self.merge(X, r, ri)

        self.log_dict({
            'epoch_cluster_loss': cluster_loss,
            'epoch_subcluster_loss': subcluster_loss,
            'cbi': cluster_burned_in,
            'mbi': cluster_m_burned_in,
            'sbi': subcluster_burned_in,
            'smbi': subcluster_m_burned_in,
        }, prog_bar=True)

    def split(self, X, r, ri):
        decisions = self.ms_rules.split_decisions(X, r, ri)
        logger.info('Split decisions: {}'.format(decisions))

        if not decisions.any():
            return

        cluster_opt: Optimizer
        cluster_opt, subcluster_opt = self.optimizers()

        # Split clustering network
        for p in self.cluster_net.out_net.parameters():
            cluster_opt.state.pop(p)

        self.cluster_net.split(decisions, self.hparams.init_weights)
        cluster_opt.param_groups[1]['params'] = list(self.cluster_net.out_net.parameters())
        self.cluster_net.to(self.device)

        # Split subcluster network
        subcluster_opt.state.clear()
        self.subcluster_net.split(decisions, self.hparams.init_weights_split_sub)
        subcluster_opt.param_groups[0]['params'] = list(self.subcluster_net.parameters())
        self.subcluster_net.to(self.device)

        # Split GMM params
        self.ms_rules.split(decisions, X, r, ri)

        self.k = self.cluster_gmm.n_components
        self.cluster_burnin.reset()
        self.cluster_m_burnin.reset()
        self.subcluster_burnin.reset()
        self.subcluster_m_burnin.reset()

        self.sanity_check()

    def merge(self, X, r, ri):
        decisions = self.ms_rules.merge_decisions(X, r)
        logger.info('Merge decisions: {}'.format(decisions))

        if len(decisions) == 0:
            return

        cluster_opt: Optimizer
        subcluster_opt: Optimizer
        cluster_opt, subcluster_opt = self.optimizers()

        # Merge clustering network
        for p in self.cluster_net.out_net.parameters():
            cluster_opt.state.pop(p)

        self.cluster_net.merge(decisions, self.hparams.init_weights)
        cluster_opt.param_groups[1]['params'] = list(self.cluster_net.out_net.parameters())
        self.cluster_net.to(self.device)

        # Merge subcluster network
        subcluster_opt.state.clear()
        self.subcluster_net.merge(decisions, self.hparams.init_weights_merge_sub)
        subcluster_opt.param_groups[0]['params'] = list(self.subcluster_net.parameters())
        self.subcluster_net.to(self.device)

        # Merge GMM params
        self.ms_rules.merge(decisions, X, r, ri)

        self.k = self.cluster_gmm.n_components
        self.cluster_burnin.reset()
        self.cluster_m_burnin.reset()
        self.subcluster_burnin.reset()
        self.subcluster_m_burnin.reset()

        self.sanity_check()

    def sanity_check(self):
        # Debug: do a few asserts
        assert self.cluster_gmm.n_components == self.k
        assert self.cluster_gmm.pi.shape == (self.k,)
        assert self.cluster_gmm.mus.shape == (self.k, self.repr_dim)
        assert self.cluster_gmm.covs.shape == (self.k, self.repr_dim, self.repr_dim)
        assert self.subcluster_gmm.n_components == self.k * self.subcluster_gmm.n_subcomponents
        for component in self.subcluster_gmm.components:
            assert component.pi.shape == (2,)
            assert component.mus.shape == (2, self.repr_dim)
            assert component.covs.shape == (2, self.repr_dim, self.repr_dim)

        assert len(self.subcluster_gmm) == self.k
        # assert self.cluster_net.in_net.weight.data.shape == (self.hparams.lat_dim, self.repr_dim)
        # assert self.cluster_net.out_net.weight.data.shape == (self.k, self.hparams.lat_dim)
        # assert self.subcluster_net.in_net.weight.data.shape == (self.hparams.lat_dim * self.k, self.repr_dim)
        # assert self.subcluster_net.out_net.weight.data.shape == (self.k * 2, self.hparams.lat_dim * self.k)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = {}
        x = batch

        r = self.forward(x)
        out['r'] = r.detach()

        if self.hparams.subcluster:
            z = r.detach().argmax(dim=-1)
            ri = self.subcluster_net(x, z)
            out['ri'] = ri.detach()

        return out

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        self.val_r = torch.cat(dicts_extract(outputs, 'r'), dim=0)
        if self.hparams.subcluster:
            self.val_ri = torch.cat(dicts_extract(outputs, 'ri'), dim=0)

    def configure_optimizers(self):
        optimizers = []

        cluster_params = torch.nn.ParameterList([
            p for n, p in self.cluster_net.named_parameters() if "out_net" not in n
        ])
        cluster_opt = torch.optim.Adam(cluster_params, lr=self.hparams.cluster_lr)
        cluster_opt.add_param_group({'params': self.cluster_net.out_net.parameters()})
        optimizers.append({"optimizer": cluster_opt})

        if self.hparams.subcluster:
            subcluster_opt = torch.optim.Adam(self.subcluster_net.parameters(), lr=self.hparams.subcluster_lr)
            optimizers.append({"optimizer": subcluster_opt})

        return optimizers

    def train_dataloader(self):
        return CustomLoader(self.dataset, shuffle=True, **self.hparams.loader_args)

    def val_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def predict_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)
