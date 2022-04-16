from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from tch_geometric.loader import CustomLoader
from torch import Tensor
from torch.utils.data import Dataset

from ml.algo.dpm.dpmm import DirichletProcessMixtureModel, InitMode
from ml.algo.dpm.dpmm_stacked import StackedDirichletProcessMixtureModel
from ml.algo.dpm.priors import DirichletPrior, DirichletParams, NIWPrior
from ml.algo.dpm.statistics import MultivarNormalParams
from ml.algo.dpm.stochastic import MeanParams
from ml.utils import HParams, Metric, OutputExtractor, DataLoaderParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class DPMMSCModelParams(HParams):
    init_k: int = 1
    subcluster: bool = True
    # subcluster: bool = False
    metric: Metric = Metric.L2

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
    validation_outputs: Dict[str, Tensor]

    def __init__(self, dataset: Dataset, hparams: DPMMSCModelParams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.dataset = dataset

        self.repr_dim = dataset[[0]].shape[-1]
        self.pi_prior = DirichletPrior(DirichletParams(hparams.prior_alpha))
        self.mu_cov_prior = NIWPrior.from_data(
            dataset[torch.arange(len(dataset), dtype=torch.long)],
            hparams.prior_nu, hparams.prior_kappa, hparams.prior_sigma_scale
        )

        self.clusters = DirichletProcessMixtureModel(
            hparams.init_k, self.repr_dim, self.hparams.metric, self.pi_prior, self.mu_cov_prior
        )
        self.cluster_mp = MeanParams(hparams.init_k, self.repr_dim)
        if self.hparams.subcluster:
            self.subclusters = StackedDirichletProcessMixtureModel(
                hparams.init_k, 2, self.repr_dim, self.hparams.metric, self.pi_prior, self.mu_cov_prior
            )
            self.subcluster_mp = MeanParams(hparams.init_k * 2, self.repr_dim)

    @property
    def k(self):
        return self.clusters.n_components

    def forward(self, X):
        out = {}

        if self.clusters.is_initialized:
            r = self.clusters.estimate_log_prob(X)
            z = r.argmax(dim=-1)
            out.update(dict(r=r, z=z))

            if self.hparams.subcluster and self.subclusters.is_initialized:
                ri = self.subclusters.estimate_log_prob(X, z)
                zi = ri.argmax(dim=-1)
                out.update(dict(ri=ri, zi=zi))

        return out

    def on_train_start(self) -> None:
        if not self.clusters.is_initialized:
            X = self.dataset[torch.arange(len(self.dataset), dtype=torch.long)]
            self.clusters.reinitialize(X, None, self.hparams.cluster_init_mode)

        if self.hparams.subcluster and not self.subclusters.is_initialized:
            X = self.dataset[torch.arange(len(self.dataset), dtype=torch.long)]
            r = self.clusters.estimate_log_prob(X)
            self.subclusters.reinitialize(X, r, self.hparams.subcluster_init_mode)

    def on_train_epoch_start(self) -> None:
        self.cluster_mp.reset()
        if self.hparams.subcluster:
            self.subcluster_mp.reset()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        X = batch

        r = self.clusters.estimate_log_prob(X)
        z = r.argmax(dim=-1)

        Ns, (mus, covs) = self.clusters.compute_params(X, r)
        self.cluster_mp.push(Ns, mus, covs)

        if self.hparams.subcluster:
            ri = self.subclusters.estimate_log_prob(X, z)
            Ns_K, (mus_K, covs_K) = self.subclusters.compute_params(X, z, ri)
            self.subcluster_mp.push(Ns_K, mus_K, covs_K)

    def training_epoch_end(self, outputs):
        logger.info("Updating cluster params")
        Ns, mus, covs = self.cluster_mp.compute()
        self.clusters.update_params(MultivarNormalParams(mus, covs), Ns)

        if self.hparams.subcluster:
            logger.info("Updating subcluster params")
            Nsi, musi, covsi = self.subcluster_mp.compute()
            self.subclusters.update_params(MultivarNormalParams(musi, covsi), Nsi)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        out = self.forward(X)
        return out

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        outputs = OutputExtractor(outputs)
        self.validation_outputs = {}

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
