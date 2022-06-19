from dataclasses import dataclass
from typing import Union, List, Optional

import torch.nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch.utils.data import Dataset

from datasets import GraphDataset
from ml.algo.dpmm.dpmsc import DPMSC, DPMSCHParams
from ml.models.base.base_model import BaseModel
from ml.models.base.clustering_datamodule import ClusteringDataModule
from ml.utils import HParams, DataLoaderParams, OptimizerParams
from ml.utils.training import ClusteringStage


@dataclass
class MGCOMComDetModelParams(DPMSCHParams):
    n_restart: int = 1


class MGCOMComDetModel(BaseModel):
    def __init__(
        self,
        hparams: MGCOMComDetModelParams,
        optimizer_params: Optional[OptimizerParams] = None,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())
        self.automatic_optimization = False
        self.phantom = torch.nn.Parameter(torch.randn(1), requires_grad=False)

        self.cluster_model = DPMSC(hparams)
        self.stage = ClusteringStage.Clustering
        self.sample_space_version = 0

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pass

    def on_train_epoch_end(self) -> None:
        self.log_dict({'k': self.cluster_model.n_components}, prog_bar=True)

    def estimate_assignment(self, X: Tensor) -> Tensor:
        return self.cluster_model.clusters.predict(X)


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
