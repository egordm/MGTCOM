from dataclasses import dataclass
from typing import Union, List, Optional, Any

import torch.nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch.utils.data import Dataset

from datasets import GraphDataset
from ml.algo.dpmm.dpmsc import DPMSCParams, DPMSC, DPMSCHParams
from ml.models.base.base_model import BaseModel
from ml.models.base.clustering_datamodule import ClusteringDataModule
from ml.utils import HParams, DataLoaderParams, OptimizerParams
from ml.utils.training import ClusteringStage


@dataclass
class MGCOMComDetModelParams(DPMSCHParams):
    pass


class MGCOMComDetModel(BaseModel):
    def __init__(
        self,
        hparams: MGCOMComDetModelParams,
        optimizer_params: Optional[OptimizerParams] = None,
        init_z=None,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())
        self.automatic_optimization = False
        self.phantom = torch.nn.Parameter(torch.randn(1), requires_grad=False)

        self.cluster_model = DPMSC(hparams)
        self.stage = ClusteringStage.Clustering
        self.sample_space_version = 0

    @property
    def k(self):
        return self.cluster_model.n_components

    @property
    def mus(self) -> Tensor:
        return self.cluster_model.clusters.mus

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X = batch
        return {'X': X.detach()}

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)

        X = self.train_outputs.extract_cat('X', cache=True)
        self.cluster_model.fit(X)

        self.log_dict({
            'k': self.k,
        }, prog_bar=True)

    def forward(self, X):
        X = X.detach()
        out = {'X': X}

        if self.cluster_model.is_fitted:
            r = self.cluster_model.estimate_log_resp(X).exp()
            z = r.argmax(dim=-1)
            out.update(dict(r=r, z=z))
            #
            # if self.cluster_model.hparams.subcluster:
            #     ri = self.cluster_model.subclusters.estimate_assignment(X, z)
            #     zi = ri.argmax(dim=-1)
            #     out.update(dict(ri=ri, zi=zi))

        return out

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
