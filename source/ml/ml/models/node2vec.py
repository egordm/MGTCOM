from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import Data

from datasets import GraphDataset
from ml.layers.loss.hinge_loss import HingeLoss
from ml.layers.loss.skipgram_loss import SkipgramLoss
from ml.models.base.clustering_mixin import ClusteringMixin, ClusteringMixinParams
from ml.models.base.feature_model import FeatureModel
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams, Node2VecSampler
from datasets.transforms.to_homogeneous import to_homogeneous

from ml.models.base.hgraph_datamodule import HomogenousGraphDataModule
from ml.utils import Metric, OptimizerParams, DataLoaderParams, HParams

EPS = 1e-15


class UnsupervisedLoss(Enum):
    HINGE = 'hinge'
    SKIPGRAM = 'skipgram'


@dataclass
class Node2VecModelParams(HParams):
    metric: Metric = Metric.DOTP
    """Metric to use for distance/similarity calculation. (for loss)"""
    loss: UnsupervisedLoss = UnsupervisedLoss.HINGE
    """Unsupervised loss function to use."""
    hinge_margin: float = 0.1
    """Margin for hinge loss."""
    hinge_adaptive: bool = False
    """Whether to use adaptive hinge loss."""


@dataclass
class Node2VecWrapperModelParams(Node2VecModelParams):
    repr_dim: int = 32
    """Dimension of the representation vectors."""


class Node2VecModel(FeatureModel):
    hparams: Union[Node2VecModelParams, OptimizerParams]

    def __init__(
            self,
            embedder: torch.nn.Module,
            hparams: Node2VecModelParams,
            optimizer_params: Optional[OptimizerParams] = None
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())

        self.embedder = embedder
        if self.hparams.loss == UnsupervisedLoss.HINGE:
            self.loss_fn = HingeLoss(self.hparams.metric,
                margin=self.hparams.hinge_margin, adaptive=self.hparams.hinge_adaptive)
        elif self.hparams.loss == UnsupervisedLoss.SKIPGRAM:
            self.loss_fn = SkipgramLoss(self.hparams.metric)

    @property
    def repr_dim(self):
        return self.embedder.repr_dim

    def forward(self, batch) -> Any:
        node_meta = batch
        return self.embedder(node_meta)

    def loss(self, pos_walks: Tensor, neg_walks: Tensor, Z: Tensor):
        return self.loss_fn(Z, pos_walks, neg_walks)

    def training_step(self, batch, batch_idx):
        pos_walks, neg_walks, node_meta = batch
        Z = self.embedder(node_meta)

        loss = self.loss(pos_walks, neg_walks, Z)
        return loss


@dataclass
class Node2VecDataModuleParams(GraphDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class Node2VecDataModule(HomogenousGraphDataModule):
    hparams: Union[Node2VecDataModuleParams, DataLoaderParams]

    def train_sampler(self, data: Data) -> Optional[Sampler]:
        return Node2VecSampler(
            data.edge_index, data.num_nodes,
            hparams=self.hparams.n2v_params
        )

    def eval_sampler(self, data: Data) -> Optional[Sampler]:
        return None


@dataclass
class Node2VecClusModelParams(Node2VecWrapperModelParams, ClusteringMixinParams):
    pass


class Node2VecClusModel(ClusteringMixin, Node2VecModel):
    def training_step(self, batch, batch_idx):
        pos_walks, neg_walks, node_meta = batch
        Z = self.embedder(node_meta)

        loss = self.loss(pos_walks, neg_walks, Z)
        return {
            'loss': loss,
            'Z': Z.detach(),
        }

