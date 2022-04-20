from dataclasses import dataclass
from typing import Union, Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor

from datasets import GraphDataset
from ml.algo.transforms import HeteroMappingTransform
from ml.data import Compose
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams, Node2VecSampler
from ml.models.base.embedding import HeteroEmbeddingModel
from ml.models.base.hgraph_datamodule import HeterogenousGraphDataModule
from ml.models.node2vec import Node2VecModel
from ml.utils import DataLoaderParams, OptimizerParams, Metric


class Het2VecModel(HeteroEmbeddingModel):
    def __init__(
            self,
            embedder: torch.nn.Module,
            metric: Metric = Metric.L2,
            optimizer_params: Optional[OptimizerParams] = None
    ) -> None:
        super().__init__(optimizer_params)

        self.embedder = embedder
        self.n2v = Node2VecModel(
            None,
            metric=metric,
            optimizer_params=optimizer_params
        )

    def forward(self, batch):
        node_idx_dict, node_perm_dict = batch
        Z_emb = self.embedder(node_idx_dict)

        return Z_emb

    def _forward_emb(self, node_meta) -> Tensor:
        node_idx_dict, node_perm_dict = node_meta
        Z_dict = self.embedder(node_idx_dict)

        # Transform hetero data to homogenous data in the sampled order
        batch_size = sum([len(v) for v in node_idx_dict.values()])
        Z = torch.zeros(batch_size, self.embedder.repr_dim, device=self.device)
        for node_type, node_perm in node_perm_dict.items():
            Z[node_perm] = Z_dict[node_type]

        return Z

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pos_walks, neg_walks, node_meta = batch

        Z_emb = self._forward_emb(node_meta)

        loss = self.n2v.loss(pos_walks, neg_walks, Z_emb)
        return loss


@dataclass
class Het2VecDataModuleParams(GraphDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class Het2VecDataModule(HeterogenousGraphDataModule):
    hparams: Union[Het2VecDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: Het2VecDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__(dataset, hparams, loader_params)
        # Since induction doesnt work on node2vec
        self.train_data, self.val_data, self.test_data = self.data, self.data, self.data

    def train_sampler(self) -> Optional[Sampler]:
        data = self.test_data  # Since induction doesnt work on node2vec
        mapper = HeteroMappingTransform(data.num_nodes_dict, to_hetero=True)

        hdata = data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)
        n2v_sampler = Node2VecSampler(hdata.edge_index, hdata.num_nodes, hparams=self.hparams.n2v_params)

        def remap_n2v_batch(batch):
            pos_walks, neg_walks, node_meta = batch
            node_meta_new = mapper(node_meta)
            return (pos_walks, neg_walks, node_meta_new)

        # noinspection PyTypeChecker
        return Compose([
            n2v_sampler,
            remap_n2v_batch
        ])

    def eval_sampler(self) -> Optional[Sampler]:
        return None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return NodesLoader(
            self.train_data.num_nodes,
            transform=self.train_sampler(),
            shuffle=True,
            **self.loader_params.to_dict()
        )


