from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor
from tch_geometric.loader.hgt_loader import HGTLoader
from tch_geometric.transforms import NegativeSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from torch_geometric.data import HeteroData
from simple_parsing import choice

from datasets import GraphDataset
from ml.data import ContrastiveTopoDataLoader, HeteroNodesDataset
from ml.layers.embedding import HGTEmbeddingModule
from ml.layers.loss.hinge_loss import HingeLoss
from ml.layers.metrics.metric_collector import MetricCollector
from ml.models.base.embedding import BaseEmbeddingModel
from ml.utils import merge_dicts, dicts_extract
from ml.utils.config import HParams, DataLoaderParams


class ValidationDataloaderIdx(Enum):
    SequentialNodes = 0


@dataclass
class TopoEmbeddingModelParams(HParams):
    repr_dim: int = 32
    hidden_dim: Optional[int] = None
    num_layers: int = 2
    num_heads: int = 2
    neigh_agg: str = 'mean'

    sim: str = choice(['cosine', 'dotp', 'euclidean'], default='euclidean')

    lr: float = 0.01

    num_neighbors: List[int] = field(default_factory=lambda: [4, 3])
    num_neg_samples: int = 3
    num_neg_tries: int = 5

    loader_args: DataLoaderParams = DataLoaderParams()


class TopoEmbeddingModel(BaseEmbeddingModel):
    hparams: TopoEmbeddingModelParams

    def __init__(self, dataset: GraphDataset, hparams: TopoEmbeddingModelParams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.dataset = dataset

        self.embedding_module = HGTEmbeddingModule(
            metadata=self.dataset.data.metadata(),
            repr_dim=self.hparams.repr_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            group=self.hparams.neigh_agg
        )

        self.loss_fn = HingeLoss(sim=self.hparams.sim)

        self.metrics = MetricCollector({
            'mean_loss': torchmetrics.MeanMetric(),
        })

    def forward(self, batch: HeteroData, *args, **kwargs) -> Any:
        return self.embedding_module(batch, *args, **kwargs)

    def step(self, batch):
        sample_data, pn_data, node_types = batch

        emb_dict = self.forward(sample_data)
        emb = torch.cat([emb_dict[node_type] for node_type in node_types], dim=0)

        loss = self.loss_fn(emb, pn_data)

        return {
            'loss': loss,
            'mean_loss': loss.detach(),
        }

    def training_step(self, batch):
        return self.step(batch)

    def on_train_batch_end(self, outputs, *args, **kwargs) -> None:
        self.metrics.update(outputs)
        # self.log_dict(outputs, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        outputs = self.metrics.compute(epoch=True, prefix='epoch_')
        self.log_dict(outputs, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return ContrastiveTopoDataLoader(
            self.dataset.data,
            neg_sampler=NegativeSamplerTransform(
                self.dataset.data, self.hparams.num_neg_samples, self.hparams.num_neg_tries
            ),
            neighbor_sampler=HGTSamplerTransform(self.dataset.data, self.hparams.num_neighbors, temporal=False),
            shuffle=True,
            **self.hparams.loader_args,
        )

    def predict_dataloader(self):
        return HGTLoader(
            HeteroNodesDataset(self.dataset.data, temporal=False),
            neighbor_sampler=HGTSamplerTransform(self.dataset.data, self.hparams.num_neighbors, temporal=False),
            shuffle=False,
            **self.hparams.loader_args,
        )

    def val_dataloader(self):
        return [
            HGTLoader(
                HeteroNodesDataset(self.dataset.data, temporal=False),
                neighbor_sampler=HGTSamplerTransform(self.dataset.data, self.hparams.num_neighbors, temporal=False),
                shuffle=False,
                **self.hparams.loader_args,
            ),
        ]
