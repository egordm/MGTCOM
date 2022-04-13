import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, Union

import pytorch_lightning as pl
import torch
from simple_parsing import choice, field
from tch_geometric.loader import CustomLoader
from torch import Tensor
from torch.utils.data import Dataset

from ml.layers.dpm import ClusteringNet, SubClusteringNet, Priors, GaussianMixtureModel, StackedGaussianMixtureModel, \
    SplitMode, MergeMode, InitMode
from ml.layers.dpm.mhsc_rules import MHSCRules
from ml.utils import dicts_extract, flat_iter
from ml.utils.config import HParams, DataLoaderParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class DPMClusteringModelParams(HParams):
    lat_dim: int = 32
    init_k: int = 1
    # init_k: int = 2
    subcluster: bool = True

    sim: str = choice(['cosine', 'dotp', 'euclidean'], default='euclidean')

    epoch_start_m: int = 10
    epoch_start_msub: int = 30

    prior_dir_counts: float = 0.1
    prior_kappa: float = 0.0001
    prior_nu: float = field(default=12.0, help="Need to be at least repr_dim + 1")
    prior_sigma_choice: str = choice(['data_std', 'isotropic'], default='data_std')
    prior_sigma_scale: float = 0.005

    mu_init_fn: InitMode = InitMode.KMeans
    mu_sub_init_fn: InitMode = InitMode.KMeans1D
    mu_update_fn: InitMode = InitMode.SoftAssignment

    alpha: float = 10.0
    split_prob: Optional[float] = None
    merge_prob: Optional[float] = None
    min_split_points: int = 6
    n_merge_neighbors: int = 3
    split_mode: SplitMode = SplitMode.Random
    merge_mode: MergeMode = MergeMode.Same

    cluster_lr: float = 0.01
    subcluster_lr: float = 0.01

    loader_args: DataLoaderParams = DataLoaderParams()


class OptimizerIdx(IntEnum):
    Cluster = 0
    SubCluster = 1


class Stage(IntEnum):
    BurnIn = 0
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
        self.subcluster_net = SubClusteringNet(self.k, self.repr_dim,
                                               self.hparams.lat_dim) if self.hparams.subcluster else None

        if self.hparams.prior_nu < repr_dim + 1:
            logging.warning("prior_nu must be at least repr_dim + 1")
            self.hparams.prior_nu = repr_dim + 1

        self.prior = Priors(
            kappa=self.hparams.prior_kappa, nu=self.hparams.prior_nu,
            sigma_scale=self.hparams.prior_sigma_scale, prior_sigma_choice=self.hparams.prior_sigma_choice,
        )
        self.cluster_gmm = GaussianMixtureModel(
            self.k, self.repr_dim, sim=self.hparams.sim, loss='kl', init_mode=self.hparams.mu_init_fn
        )
        self.subcluster_gmm = StackedGaussianMixtureModel(
            self.k, 2, self.repr_dim, sim=self.hparams.sim, loss='iso', init_mode=self.hparams.mu_sub_init_fn
        ) if self.hparams.subcluster else None

        self.ms_rules = MHSCRules(
            self.prior, self.cluster_gmm, self.subcluster_gmm,
            self.hparams.alpha, self.hparams.split_prob, self.hparams.merge_prob,
            self.hparams.min_split_points, self.hparams.n_merge_neighbors,
            sim=self.hparams.sim,
        )

        self.stage = Stage.BurnIn
        self.last_action = Action.NoAction
        # self.last_action = Action.Split
        self.action = Action.NoAction

    def forward(self, batch: Tensor):
        return self.cluster_net(batch)

    def on_train_start(self):
        # Initialize GMM on the data
        xs = self.dataset[torch.arange(len(self.dataset), dtype=torch.long)]
        self.prior.init_priors(xs)

        logger.info(f"Initializing cluster params")
        self.cluster_gmm.reinit_params(xs, self.prior)

    def on_train_epoch_start(self) -> None:
        if self.stage == Stage.BurnIn:
            if self.current_epoch >= self.hparams.epoch_start_m:
                self.stage = Stage.Clustering

        if self.stage == Stage.Clustering:
            if self.hparams.subcluster and self.current_epoch >= self.hparams.epoch_start_msub:
                self.stage = Stage.SubClustering

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        out = {}
        x = batch
        r = self.cluster_net(x)

        cluster_loss = self.cluster_gmm.e_step(x, r)
        out['cluster_loss'] = cluster_loss.detach()

        if optimizer_idx == OptimizerIdx.Cluster:
            loss = cluster_loss

            out.update({
                'r': r.detach(),
                'x': x.detach(),
            })
        elif optimizer_idx == OptimizerIdx.SubCluster and self.is_subclusters_initialized():
            r = r.detach()
            z = r.argmax(dim=-1)
            ri = self.subcluster_net(x, z)

            subcluster_loss = self.subcluster_gmm.e_step(x, ri)
            out['subcluster_loss'] = cluster_loss.detach()

            loss = subcluster_loss

            out.update({
                'ri': ri.detach(),
            })
        else:
            return None

        out['loss'] = loss
        return out

    def training_epoch_end(self, outputs) -> None:
        X = torch.cat(dicts_extract(flat_iter(outputs), 'x'), dim=0)
        r = torch.cat(dicts_extract(flat_iter(outputs), 'r'), dim=0)

        # Clustering M step
        if self.stage >= Stage.Clustering:
            logger.info("Updating cluster params")
            self.cluster_gmm.m_step(X, r, self.prior)

        # Initialize subcluster GMM (later in the training)
        if self.hparams.subcluster and self.current_epoch + 1 == self.hparams.epoch_start_msub:
            logger.info("Initializing subcluster params")
            self.subcluster_gmm.reinit_params(X, r, self.prior)

        # Subcluster M step
        if self.stage >= Stage.SubClustering:
            logger.info("Updating subcluster params")
            ri = torch.cat(dicts_extract(flat_iter(outputs), 'ri'), dim=0)
            self.subcluster_gmm.m_step(X, r, ri, self.prior)

        # Decide on a Merge or Split action
        self.action = Action.NoAction
        if self.stage >= Stage.SubClustering:
            if self.last_action != Action.Split:
                self.action = Action.Split
            elif self.last_action != Action.Merge:
                self.action = Action.Merge

        # Perform the action
        if self.action == Action.Split:
            self.last_action = Action.Split
            self.split(X, r, ri)

        elif self.action == self.action.Merge:
            self.last_action = Action.Merge
            self.merge(X, r, ri)

        self.log_dict({
            'stage_burnin': self.stage == Stage.BurnIn,
            'stage_clustering': self.stage == Stage.Clustering,
            'stage_subclustering': self.stage == Stage.SubClustering,
        })

    def split(self, X, r, ri):
        decisions = self.ms_rules.split_decisions(X, r, ri)
        logger.info('Split decisions: {}'.format(decisions))

        if not decisions.any():
            return

        cluster_opt, subcluster_opt = self.optimizers()

        # Split clustering network
        for p in self.cluster_net.out_net.parameters():
            cluster_opt.state.pop(p)

        self.cluster_net.split(decisions, SplitMode.Same)
        cluster_opt.param_groups[1]['params'] = list(self.cluster_net.out_net.parameters())
        self.cluster_net.out_net.to(self.device)

        # Split subcluster network
        for p in self.subcluster_net.parameters():
            subcluster_opt.state.pop(p)

        self.subcluster_net.split(decisions, SplitMode.Same)
        subcluster_opt.param_groups[0]['params'] = list(self.subcluster_net.parameters())

        # Split GMM params
        self.ms_rules.split(decisions, X, r, ri)

        self.k = self.cluster_gmm.n_components

        self.sanity_check()

    def merge(self, X, r, ri):
        decisions = self.ms_rules.merge_decisions(X, r)
        logger.info('Merge decisions: {}'.format(decisions))

        if len(decisions) == 0:
            return

        cluster_opt, subcluster_opt = self.optimizers()

        # Merge clustering network
        for p in self.cluster_net.out_net.parameters():
            cluster_opt.state.pop(p)

        self.cluster_net.merge(decisions, MergeMode.Same)
        cluster_opt.param_groups[1]['params'] = list(self.cluster_net.out_net.parameters())
        self.cluster_net.out_net.to(self.device)

        # Merge subcluster network
        for p in self.subcluster_net.parameters():
            subcluster_opt.state.pop(p)

        self.subcluster_net.merge(decisions, MergeMode.Same)
        subcluster_opt.param_groups[0]['params'] = list(self.subcluster_net.parameters())

        # Merge GMM params
        self.ms_rules.merge(decisions, X, r, ri)

        self.k = self.cluster_gmm.n_components

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
        assert self.cluster_net.in_net.weight.data.shape == (self.hparams.lat_dim, self.repr_dim)
        assert self.cluster_net.out_net.weight.data.shape == (self.k, self.hparams.lat_dim)
        assert self.subcluster_net.in_net.weight.data.shape == (self.hparams.lat_dim * self.k, self.repr_dim)
        assert self.subcluster_net.out_net.weight.data.shape == (self.k * 2, self.hparams.lat_dim * self.k)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = {}
        x = batch

        r = self.forward(x)
        out['r'] = r.detach()

        z = r.detach().argmax(dim=-1)
        ri = self.subcluster_net(x, z)
        out['ri'] = ri.detach()

        return out

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        self.val_r = torch.cat(dicts_extract(outputs, 'r'), dim=0)
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
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def val_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def predict_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def is_subclusters_initialized(self):
        return self.hparams.subcluster and self.current_epoch >= self.hparams.epoch_start_msub
