from typing import Any, List, Union, Dict, Tuple

import pytorch_lightning as pl
import torch.nn
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from tch_geometric.loader.budget_loader import BudgetLoader
from tch_geometric.loader.hgt_loader import HGTLoader
from tch_geometric.transforms import NegativeSamplerTransform
from tch_geometric.transforms.budget_sampling import BudgetSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from ml.layers import HingeLoss, NegativeEntropyRegularizer, Compose
from ml.layers.embedding import HeteroEmbeddingModule
from ml.layers.metrics import MetricBag
from ml.loaders import ContrastiveDataLoader, HeteroEdgesDataset, HeteroNodesDataset
from ml.loaders.temporal_sampling import TemporalSamplerLoader
from ml.utils import merge_dicts


class TemporalModel(pl.LightningModule):
    def __init__(
            self,
            embedding_module: HeteroEmbeddingModule,
            temp_embedding_module: HeteroEmbeddingModule,
            clustering_module: torch.nn.Module,
            temp_clustering_module: torch.nn.Module,
            c_weight: float = 1.0,
            ne_weight: float = 0.001,
            lr: float = 0.01,
            lr_cosine: bool = False,
            use_clustering: bool = False,
            sim='dotp',
            temp_agg = 'cat',
            *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_module = embedding_module
        self.temp_embedding_module = temp_embedding_module

        self.clustering_module = clustering_module
        self.temp_clustering_module = temp_clustering_module

        self.ne_weight = ne_weight
        self.c_weight = c_weight
        self.lr = lr
        self.lr_cosine = lr_cosine
        self.use_clustering = use_clustering

        self.positional_loss = HingeLoss(sim=sim)
        self.clustering_loss = HingeLoss(sim=sim)
        self.ne_loss = NegativeEntropyRegularizer()

        self.temp_agg = temp_agg
        if self.temp_agg == 'cat':
            self.temp_agg_fn = lambda xs: torch.cat(xs, dim=-1)
        elif self.temp_agg == 'sum':
            self.temp_agg_fn = sum
        else:
            raise ValueError(f'Unknown aggregation {self.agg}')

        self.metrics = MetricBag({
            'loss': torchmetrics.MeanMetric(),
            'p_loss': torchmetrics.MeanMetric(),
            'c_loss': torchmetrics.MeanMetric(),
        })

    def forward(self, batch: HeteroData, *args, **kwargs) -> Any:
        top_emb = self.embedding_module(batch)
        temp_emb = self.temp_embedding_module(batch)
        emb = merge_dicts([top_emb, temp_emb], self.temp_agg_fn)

        return emb

    def step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        sample_data, pn_data, node_types = batch

        p_top_emb_dict = self.embedding_module(sample_data)
        p_top_emb = torch.cat([p_top_emb_dict[node_type] for node_type in node_types], dim=0)
        p_tmp_emb_dict = self.temp_embedding_module(sample_data)
        p_tmp_emb = torch.cat([p_tmp_emb_dict[node_type] for node_type in node_types], dim=0)

        p_emb = self.temp_agg_fn([p_top_emb, p_tmp_emb])
        p_loss = self.positional_loss(p_emb, pn_data)

        if self.use_clustering:
            centroids = self.centroids()
            c_emb = self.clustering_module.sim_fn(p_emb.unsqueeze(1), centroids.unsqueeze(0)).softmax(dim=-1)

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

    def centroids(self) -> torch.Tensor:
        return self.temp_agg_fn([
            self.clustering_module.centroids.weight.data,
            self.temp_clustering_module.centroids.weight.data,
        ])

    def compute_soft_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.clustering_module\
            .sim_fn(embeddings.unsqueeze(1), self.centroids().unsqueeze(0))\
            .softmax(dim=-1)

    def compute_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.compute_soft_assignments(embeddings).argmax(dim=-1)


class TemporalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data: HeteroData,
            num_samples: Union[List[int], Dict[NodeType, List[int]]],
            window: Tuple[int, int],
            num_neg_samples: int = 3,
            repeat_count: int = 1,
            batch_size: int = 16,
            num_workers: int = 0,
            prefetch_factor=2,
            persistent_workers=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data = data
        self.window = window
        self.num_neg_samples = num_neg_samples
        self.repeat_count = repeat_count
        self.loader_args = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.neighbor_sampler = HGTSamplerTransform(data, num_samples, temporal=False)
        # self.neighbor_sampler = BudgetSamplerTransform(data, num_samples, self.window, forward=False, relative=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return TemporalSamplerLoader(
            self.data,
            window=self.window,
            num_neg=self.num_neg_samples, repeat_count=self.repeat_count,
            neighbor_sampler=self.neighbor_sampler,
            shuffle=True,
            **self.loader_args,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return HGTLoader(
        # return BudgetLoader(
            HeteroNodesDataset(self.data, temporal=False),
            neighbor_sampler=self.neighbor_sampler,
            shuffle=False,
            **self.loader_args,
        )
