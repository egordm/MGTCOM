from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from datasets import GraphDataset
from ml.models.base.feature_model import FeatureModel
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams, Node2VecSampler
from ml.data.transforms.to_homogeneous import to_homogeneous

from ml.models.base.hgraph_datamodule import HomogenousGraphDataModule
from ml.utils import Metric, OptimizerParams, DataLoaderParams

EPS = 1e-15


class Node2VecModel(FeatureModel):
    def __init__(
            self,
            embedder: torch.nn.Module,
            metric: Metric = Metric.DOTP,
            hparams: Any = None,
            optimizer_params: Optional[OptimizerParams] = None
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams)

        self.embedder = embedder
        self.metric = metric
        self.sim_fn = metric.pairwise_sim_fn

    @property
    def repr_dim(self):
        return self.embedder.repr_dim

    def forward(self, batch) -> Any:
        node_meta = batch
        return self.embedder(node_meta)

    def loss(self, pos_walks: Tensor, neg_walks: Tensor, Z: Tensor):
        pos_walks_Z = Z[pos_walks.view(-1)].view(*pos_walks.shape, Z.shape[-1])
        p_head, p_rest = pos_walks_Z[:, 0].unsqueeze(dim=1), pos_walks_Z[:, 1:]
        p_sim = self.sim_fn(p_head, p_rest).view(-1)
        # p_head_, p_rest_ = F.normalize(p_head, p=2, dim=2), F.normalize(p_rest, p=2, dim=2)
        # p_sim = (p_head_ * p_rest_).sum(dim=-1).view(-1)  # Always use dot product
        # p_sim = -1 * (p_head - p_rest).pow(2).sum(dim=-1).view(-1)  # L2
        pos_loss = -torch.log(torch.sigmoid(p_sim) + EPS).mean()

        neg_walks_Z = Z[neg_walks.view(-1)].view(*neg_walks.shape, Z.shape[-1])
        n_head, n_rest = neg_walks_Z[:, 0].unsqueeze(dim=1), neg_walks_Z[:, 1:]
        n_sim = self.sim_fn(n_head, n_rest).view(-1)
        # n_head_, n_rest_ = F.normalize(n_head, p=2, dim=2), F.normalize(n_rest, p=2, dim=2)
        # n_sim = (n_head_ * n_rest_).sum(dim=-1).view(-1)  # Always use dot product
        # n_sim = -1 * (n_head - n_rest).pow(2).sum(dim=-1).view(-1)  # L2
        neg_loss = -torch.log(1 - torch.sigmoid(n_sim) + EPS).mean()

        return pos_loss + neg_loss

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

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: Node2VecDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__(dataset, hparams, loader_params)
        hdata = to_homogeneous(self.data)
        self.train_data, self.val_data, self.test_data = hdata, hdata, hdata  # Since induction doesnt work on node2vec

    def train_sampler(self) -> Optional[Sampler]:
        return Node2VecSampler(
            self.train_data.edge_index, self.train_data.num_nodes,
            hparams=self.hparams.n2v_params
        )

    def eval_sampler(self) -> Optional[Sampler]:
        return None
