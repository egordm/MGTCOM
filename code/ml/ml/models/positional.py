from typing import Any, List, Union, Dict

import pytorch_lightning as pl
import torch.nn
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from tch_geometric.loader.hgt_loader import HGTLoader
from tch_geometric.transforms import NegativeSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from ml.layers import HingeLoss, NegativeEntropyRegularizer
from ml.layers.embedding import HeteroEmbeddingModule
from ml.layers.metrics import MetricBag
from ml.loaders import ContrastiveDataLoader, HeteroEdgesDataset, HeteroNodesDataset
from ml.utils import merge_dicts


class PositionalModel(pl.LightningModule):
    def __init__(
            self,
            embedding_module: HeteroEmbeddingModule,
            clustering_module: torch.nn.Module,
            c_weight: float = 1.0,
            ne_weight: float = 0.001,
            lr: float = 0.01,
            lr_cosine: bool = False,
            use_clustering: bool = False,
            sim='dotp',
            *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_module = embedding_module
        self.clustering_module = clustering_module

        self.ne_weight = ne_weight
        self.c_weight = c_weight
        self.lr = lr
        self.lr_cosine = lr_cosine
        self.use_clustering = use_clustering

        self.positional_loss = HingeLoss(sim=sim)
        self.clustering_loss = HingeLoss(sim=sim)
        self.ne_loss = NegativeEntropyRegularizer()

        self.metrics = MetricBag({
            'loss': torchmetrics.MeanMetric(),
            'p_loss': torchmetrics.MeanMetric(),
            'c_loss': torchmetrics.MeanMetric(),
        })

    def forward(self, batch: HeteroData, *args, **kwargs) -> Any:
        return self.embedding_module(batch, *args, **kwargs)

    def step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        sample_data, pn_data, node_types = batch

        p_emb_dict = self.embedding_module(sample_data)
        p_emb = torch.cat([p_emb_dict[node_type] for node_type in node_types], dim=0)

        p_loss = self.positional_loss(p_emb, pn_data)

        if self.use_clustering:
            c_emb = self.clustering_module(p_emb)

            c_loss = self.clustering_loss(c_emb, pn_data)
            ne = self.ne_loss(c_emb)

            return {
                'loss': p_loss + (c_loss * self.c_weight) + (ne * self.ne_weight),
                'p_loss': p_loss.detach(),
                'c_loss': c_loss.detach(),
                'ne_loss': ne.detach(),
            }
        else:
            return {
                'loss': p_loss,
                'p_loss': p_loss.detach(),
            }

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.step(*args, **kwargs)

    def on_train_batch_end(self, outputs, *args, **kwargs) -> None:
        self.metrics.update(outputs)
        self.log_dict(outputs, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        super().on_train_batch_end(outputs, *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        outputs = self.metrics.compute(epoch=True, prefix='epoch_')
        self.log_dict(outputs, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, eta_min=0.0001, T_0=200, T_mult=2
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def compute_embeddings(self, trainer: pl.Trainer, loader: pl.LightningDataModule):
        pred = trainer.predict(self, loader)
        return merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))


class PositionalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data: HeteroData,
            num_samples: Union[List[int], Dict[NodeType, List[int]]],
            num_neg_samples: int = 3,
            num_neg_tries: int = 5,
            batch_size: int = 16,
            num_workers: int = 0,
            prefetch_factor=2,
            persistent_workers=False,
            temporal: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data = data
        self.temporal = temporal
        self.loader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.neg_sampler = NegativeSamplerTransform(data, num_neg_samples, num_neg_tries)
        self.neighbor_sampler = HGTSamplerTransform(data, num_samples, temporal=temporal)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return ContrastiveDataLoader(
            HeteroEdgesDataset(self.data, temporal=self.temporal),
            self.neg_sampler,
            neighbor_sampler=self.neighbor_sampler,
            shuffle=True,
            **self.loader_args,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return HGTLoader(
            HeteroNodesDataset(self.data, temporal=self.temporal),
            neighbor_sampler=self.neighbor_sampler,
            shuffle=False,
            **self.loader_args,
        )
