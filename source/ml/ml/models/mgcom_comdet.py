from dataclasses import dataclass
from enum import IntEnum
from typing import Union, List, Optional, Any

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from scipy import linalg
from sklearn.mixture import BayesianGaussianMixture
from torch import Tensor
from torch.utils.data import Dataset
from sklearn import mixture
import matplotlib as mpl

from datasets import GraphDataset
from ml.algo.dpm.dpmm_sc import DPMMSCModelParams, DPMMSCModel
from ml.models.base.base_model import BaseModel
from ml.models.base.clustering_datamodule import ClusteringDataModule
from ml.utils import HParams, DataLoaderParams, OptimizerParams
from ml.utils.plot import create_colormap


class Stage(IntEnum):
    GatherSamples = 0
    Clustering = 1


@dataclass
class MGCOMComDetModelParams(DPMMSCModelParams):
    pass


class MGCOMComDetModel(BaseModel):
    def __init__(
            self,
            repr_dim: int,
            hparams: MGCOMComDetModelParams,
            optimizer_params: Optional[OptimizerParams] = None,
            init_z: Optional[Tensor] = None,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())
        self.automatic_optimization = False

        self.dpmm_model = DPMMSCModel(repr_dim, hparams)
        self.init_z = init_z
        self.stage = Stage.GatherSamples
        self.sample_space_version = 0

    @property
    def k(self):
        return self.dpmm_model.k

    @property
    def is_done(self) -> bool:
        return self.dpmm_model.is_done

    @property
    def mus(self) -> Tensor:
        return self.dpmm_model.clusters.mus

    def on_train_start(self) -> None:
        super().on_train_start()
        self.dpmm_model.to(device=self.device)

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:
        if self.is_done:
            return -1

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X = batch
        if self.stage == Stage.Clustering:
            self.dpmm_model.step_e(X)

        return {
            'X': X.detach()
        }

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)

        if self.stage == Stage.GatherSamples:
            X = self.train_outputs.extract_cat('X', cache=True)
            self.dpmm_model.reinitialize(X, incremental=True, z=self.init_z)
            self.stage = Stage.Clustering
        elif self.stage == Stage.Clustering:
            X = self.train_outputs.extract_cat('X', cache=True)
            dpgmm = BayesianGaussianMixture(
                weight_concentration_prior=1e2,
                weight_concentration_prior_type="dirichlet_distribution",
                n_components=1,
                covariance_type='full',
                init_params="kmeans",
                max_iter=1500,
            )
            dpgmm.fit(X.numpy())

            Y_ = dpgmm.predict(X)
            _min, _max = X.min(dim=0).values, X.max(dim=0).values
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            colors = create_colormap(len(dpgmm.means_))
            for i, (mean, covar, color) in enumerate(zip(dpgmm.means_, dpgmm.covariances_, colors)):
                v, w = linalg.eigh(covar)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)

            ax.set_xlim([float(_min[0]), float(_max[0])])
            ax.set_ylim([float(_min[1]), float(_max[1])])
            plt.xticks(())
            plt.yticks(())
            plt.show()

            self.dpmm_model.step_m()
            self.stage = Stage.GatherSamples if not self.dpmm_model.is_initialized else Stage.Clustering

        self.log_dict({
            'k': self.k,
        }, prog_bar=True)

    def forward(self, X):
        X = X.detach()
        out = {'X': X}

        if self.dpmm_model.is_initialized:
            r = self.dpmm_model.clusters.estimate_assignment(X)
            z = r.argmax(dim=-1)
            out.update(dict(r=r, z=z))

            if self.dpmm_model.hparams.subcluster:
                ri = self.dpmm_model.subclusters.estimate_assignment(X, z)
                zi = ri.argmax(dim=-1)
                out.update(dict(ri=ri, zi=zi))

        return out

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.forward(batch)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.forward(batch)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.forward(batch)

    def estimate_assignment(self, X: Tensor) -> Tensor:
        return self.dpmm_model.clusters.estimate_assignment(X)


@dataclass
class MGCOMComDetDataModuleParams(HParams):
    pass


class MGCOMComDetDataModule(ClusteringDataModule):
    dataset: Dataset

    def __init__(
            self,
            dataset: Dataset,
            graph_dataset: Optional[GraphDataset],
            hparams: MGCOMComDetDataModuleParams,
            loader_params: DataLoaderParams
    ):
        super().__init__(dataset, graph_dataset, loader_params)
        self.save_hyperparameters(hparams.to_dict())
