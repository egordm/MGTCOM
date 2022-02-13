from typing import Tuple

import torch
import torchmetrics
from torch_geometric.data import HeteroData

import ml
from experiments.losses.clustering import ClusterCohesionLoss, NegativeEntropyRegularizer
from experiments.models.clustering import ClusteringModule
from experiments.models.embedding import LinkPredictionModule, EmbeddingModule


class EmbeddingNet(ml.BaseModule):
    def __init__(self, embedding_module: EmbeddingModule, params: dict = None) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.params = params or {}

    def forward(self, batch: torch.Tensor):
        return self.embedding_module(batch)


class LinkPredictionNet(ml.BaseModule):
    def __init__(self, predictor_module: LinkPredictionModule, params: dict = None) -> None:
        super().__init__()
        self.predictor_module = predictor_module
        self.params = params or {}

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: torch.Tensor):
        return self.predictor_module.embedding_module(batch)

    def _step(self, batch: Tuple[HeteroData, HeteroData, torch.Tensor]):
        _, _, label = batch
        logits, dist, emb_l, emb_r = self.predictor_module(batch)
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


class SelfSupervisedClusteringNet(ml.BaseModule):
    def __init__(
            self,
            predictor_module: LinkPredictionModule,
            clustering_module: ClusteringModule,
            params: dict = None
    ) -> None:
        super().__init__()
        self.predictor_module = predictor_module
        self.clustering_module = clustering_module
        self.params = params or {}

        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.cc_loss_fn = ClusterCohesionLoss()
        self.ne_loss_fn = NegativeEntropyRegularizer()

    def forward(self, batch: torch.Tensor):
        emb = self.predictor_module.embedding_module(batch)
        q = self.clustering_module(emb)
        return emb, q

    def _step(self, batch: torch.Tensor):
        _, _, label = batch
        logits, dist, emb_l, emb_r = self.predictor_module(batch)
        he_loss = self.ce_loss_fn(logits, label)

        q_l = self.clustering_module(emb_l)
        q_r = self.clustering_module(emb_r)
        cc_loss = self.cc_loss_fn(q_l, q_r, label)
        ne = self.ne_loss_fn(q_l, q_r)

        loss = he_loss + cc_loss * self.params.get('cc_weight', 2.0) + ne * self.params.get('ne_weight', 0.01)

        pred = logits.argmax(dim=-1)
        return {
            'loss': loss,
            'he_loss': he_loss.detach(),
            'cc_loss': cc_loss.detach(),
            'ne': ne.detach(),
            'accuracy': (pred.detach(), label),
        }

    def configure_metrics(self):
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'he_loss': (torchmetrics.MeanMetric(), True),
            'cc_loss': (torchmetrics.MeanMetric(), True),
            'ne': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
        }
