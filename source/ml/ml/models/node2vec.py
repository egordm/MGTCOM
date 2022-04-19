from typing import Any, Optional

import torch
from torch import Tensor

from ml.models.base.embedding import BaseEmbeddingModel
from ml.utils import Metric, OptimizerParams

EPS = 1e-15


class Node2VecModel(BaseEmbeddingModel):
    def __init__(
            self,
            embedder: torch.nn.Module,
            metric: Metric = Metric.L2,
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
        pos_loss = -torch.log(torch.sigmoid(p_sim) + EPS).mean()

        neg_walks_Z = Z[neg_walks.view(-1)].view(*neg_walks.shape, Z.shape[-1])
        n_head, n_rest = neg_walks_Z[:, 0].unsqueeze(dim=1), neg_walks_Z[:, 1:]
        n_sim = self.sim_fn(n_head, n_rest).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(n_sim) + EPS).mean()

        return pos_loss + neg_loss

    def training_step(self, batch, batch_idx):
        pos_walks, neg_walks, node_meta = batch
        Z = self.embedder(node_meta)

        loss = self.loss(pos_walks, neg_walks, Z)
        return loss
