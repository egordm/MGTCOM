from abc import abstractmethod
from typing import Tuple

import torch
import torchmetrics

from torch_geometric.typing import Metadata
import torch_geometric.nn as tg_nn
from torch_geometric.data import HeteroData
import pytorch_lightning as pl

from experiments.models.base import BaseModule


class EmbeddingModule(pl.LightningModule):
    def __init__(self, metadata: Metadata, repr_dim: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.n_layers = n_layers

    @abstractmethod
    def forward(self, batch: HeteroData) -> torch.Tensor:
        pass


class GraphSAGEHeteroModule(EmbeddingModule):
    def __init__(self, repr_dim: int = 32, n_layers: int = 2,
                 normalize=False) -> None:
        super().__init__(None, repr_dim, n_layers)

        components = []
        for i in range(n_layers - 1):
            components.extend([
                (tg_nn.SAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
                torch.nn.ReLU(inplace=True)
            ])

        embedding_module = tg_nn.Sequential('x, edge_index', [
            *components,
            (tg_nn.SAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
        ])
        self.module = embedding_module

    def forward(self, batch: HeteroData, *args) -> torch.Tensor:
        batch_size = batch.batch_size
        return self.module(batch.x, batch.edge_index)[:batch_size]


class GraphSAGEModule(EmbeddingModule):
    def __init__(self, node_type: str, metadata: Metadata, repr_dim: int = 32, n_layers: int = 2,
                 normalize=False) -> None:
        super().__init__(metadata, repr_dim, n_layers)
        self.node_type = node_type

        components = []
        for i in range(n_layers - 1):
            components.extend([
                (tg_nn.SAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
                torch.nn.ReLU(inplace=True)
            ])

        embedding_module = tg_nn.Sequential('x, edge_index', [
            *components,
            (tg_nn.SAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
        ])
        self.module = tg_nn.to_hetero(embedding_module, metadata, aggr='mean')

    def forward(self, batch: HeteroData) -> torch.Tensor:
        batch_size = batch[self.node_type].batch_size
        return self.module(batch.x_dict, batch.edge_index_dict)[self.node_type][:batch_size]


class LinkPredictionModule(BaseModule):
    def __init__(self, embedding_module: EmbeddingModule, dist='cosine'):
        super().__init__()
        self.embedding_module = embedding_module

        if dist == 'cosine':
            self.dist = torch.nn.CosineSimilarity(dim=1)
        elif dist == 'euclidean':
            self.dist = torch.nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f'Unknown distance {dist}')

        self.lin = torch.nn.Linear(1, 2)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Tuple[HeteroData, HeteroData, torch.Tensor]):
        batch_l, batch_r, label = batch

        emb_l = self.embedding_module(batch_l)
        emb_r = self.embedding_module(batch_r)
        dist = self.dist(emb_l, emb_r)
        logits = self.lin(torch.unsqueeze(dist, 1))

        return logits, dist, emb_l, emb_r

    def _step(self, batch: Tuple[HeteroData, HeteroData, torch.Tensor]):
        _, _, label = batch
        logits, dist, emb_l, emb_r = self.forward(batch)
        loss = self.ce_loss(logits, label)

        pred = logits.argmax(dim=-1)
        return {
            'loss': loss,
            'accuracy': (pred.detach(), label),
        }

    def configure_metrics(self):
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
        }
