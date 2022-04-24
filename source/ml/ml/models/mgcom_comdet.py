from enum import IntEnum

import torch
from dataclasses import dataclass
from typing import Union, List, Optional, Any

from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch.utils.data import Dataset

from datasets import GraphDataset
from ml.algo.dpm.dpmm_sc import DPMMSCModelParams, DPMMSCModel
from ml.models.base.base_model import BaseModel
from ml.models.base.clustering_datamodule import ClusteringDataModule
from ml.utils import HParams, DataLoaderParams, OptimizerParams


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

    def training_step(self, batch) -> STEP_OUTPUT:
        X = batch
        if self.stage == Stage.Clustering:
            self.dpmm_model.step_e(X)

        return {
            'X': X
        }

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:
        if self.dpmm_model.is_done:
            return -1

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)

        if self.stage == Stage.GatherSamples:
            X = self.train_outputs.extract_cat('X')
            self.dpmm_model.reinitialize(X, incremental=True, z=self.init_z)
            self.stage = Stage.Clustering
        elif self.stage == Stage.Clustering:
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
