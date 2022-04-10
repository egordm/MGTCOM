import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tch_geometric.loader import CustomLoader
from torch import Tensor
from torch.utils.data import Dataset
from simple_parsing import choice, field

from ml.layers.dpm import ClusteringNet, SubClusteringNet
from ml.layers.dpm.gmm import GMMModule
from ml.layers.dpm.priors import Priors
from ml.utils import dicts_extract
from ml.utils.config import HParams, DataLoaderParams


@dataclass
class DPMClusteringModelParams(HParams):
    lat_dim: int = 32
    # init_k: int = 2
    init_k: int = 5

    sim: str = choice(['cosine', 'dotp', 'euclidean'], default='euclidean')

    epoch_start_m: int = 10

    prior_dir_counts: float = 0.1
    prior_kappa: float = 0.0001
    prior_nu: float = field(default=12.0, help="Need to be at least repr_dim + 1")
    prior_sigma_scale: float = 0.005

    mu_init_fn: str = choice('kmeans', 'soft_assign', 'kmeans_1d', default='kmeans')
    mu_sub_init_fn: str = choice('kmeans', 'soft_assign', 'kmeans_1d', default='kmeans_1d')
    mu_update_fn: str = choice('kmeans', 'soft_assign', default='soft_assign')

    cluster_lr: float = 0.01
    subcluster_lr: float = 0.01

    loader_args: DataLoaderParams = DataLoaderParams()


class TrainingStage(Enum):
    TrainClustering = 0


class OptimizerIdx(Enum):
    Cluster = 0
    SubCluster = 1


class DPMClusteringModel(pl.LightningModule):
    hparams: DPMClusteringModelParams
    stage: TrainingStage

    val_r: Tensor

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
        self.subcluster_net = SubClusteringNet(self.k, self.repr_dim, self.hparams.lat_dim)

        if self.hparams.prior_nu < repr_dim + 1:
            logging.warning("prior_nu must be at least repr_dim + 1")
            self.hparams.prior_nu = repr_dim + 1

        self.prior = Priors(
            K=self.k, pi_counts=self.hparams.prior_dir_counts,
            kappa=self.hparams.prior_kappa, nu=self.hparams.prior_nu, sigma_scale=self.hparams.prior_sigma_scale,
        )
        self.gmm = GMMModule(self.k, self.repr_dim, sim=self.hparams.sim)

    def forward(self, batch: Tensor):
        return self.cluster_net(batch)

    def on_train_start(self):
        # Initialize GMM on the data
        xs = self.dataset[torch.arange(len(self.dataset), dtype=torch.long)]
        self.prior.init_priors(xs)
        self.gmm.initialize(xs, None, self.prior, mode=self.hparams.mu_init_fn)

        self.stage = TrainingStage.TrainClustering

    def on_train_epoch_start(self) -> None:
        # self.stage = TrainingStage.TrainClustering
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        optimizer_idx = OptimizerIdx(optimizer_idx)
        x = batch
        r = self.cluster_net(x)

        cluster_loss = self.gmm.e_step(x, r)
        loss = cluster_loss

        if optimizer_idx == OptimizerIdx.SubCluster:
            pass

        out = {
            'loss': loss,
            'cluster_loss': cluster_loss.detach(),
        }

        if optimizer_idx == OptimizerIdx.Cluster:
            out.update({
                'r': r.detach(),
                'x': x.detach(),
            })

        return out

    def training_epoch_end(self, outputs) -> None:
        if self.stage == TrainingStage.TrainClustering:
            xs = torch.cat(dicts_extract(outputs, 'x'), dim=0)
            r = torch.cat(dicts_extract(outputs, 'r'), dim=0)

            update_params = self.current_epoch > self.hparams.epoch_start_m
            if update_params:
                self.gmm.m_step(xs, r, self.prior)

            self.log_dict({
                'update_params': update_params,
            })
        else:
            raise NotImplementedError()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        r = self.forward(batch)

        return {
            'r': r.detach(),
        }

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        self.val_r = torch.cat(dicts_extract(outputs, 'r'), dim=0)

    def configure_optimizers(self):
        # TODO: split paramerters over optimizers
        cluster_params = torch.nn.ParameterList(
            [p for n, p in self.cluster_net.named_parameters() if "out_net" not in n])
        cluster_opt = torch.optim.Adam(cluster_params, lr=self.hparams.cluster_lr)
        cluster_opt.add_param_group({'params': self.cluster_net.out_net.parameters()})

        subcluster_opt = torch.optim.Adam(self.subcluster_net.parameters(), lr=self.hparams.subcluster_lr)

        return [
            {"optimizer": cluster_opt},
            # {"optimizer": subcluster_opt},
        ]

    def train_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def val_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)

    def predict_dataloader(self):
        return CustomLoader(self.dataset, **self.hparams.loader_args)
