from dataclasses import dataclass
from typing import Union, Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import HeteroData

from datasets import GraphDataset
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams, Node2VecSampler
from ml.models.base.feature_model import HeteroFeatureModel
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.base.hgraph_datamodule import HeteroGraphDataModule
from ml.models.node2vec import Node2VecModel
from ml.utils import DataLoaderParams, OptimizerParams, Metric


class Het2VecModel(HeteroFeatureModel):
    def __init__(
            self,
            embedder: torch.nn.Module,
            metric: Metric = Metric.DOTP,
            optimizer_params: Optional[OptimizerParams] = None
    ) -> None:
        super().__init__(optimizer_params)

        self.embedder = embedder
        self.n2v = Node2VecModel(
            None,
            metric=metric,
            optimizer_params=optimizer_params
        )

    @property
    def repr_dim(self):
        return self.embedder.repr_dim

    def forward(self, batch):
        node_idx_dict, node_perm_dict = batch
        Z_emb = self.embedder(node_idx_dict)
        return Z_emb

    def forward_emb_flat(self, node_meta) -> Tensor:
        node_idx_dict, node_perm_dict = node_meta
        Z_dict = self.embedder(node_idx_dict)

        Z = ToHeteroMappingTransform.inverse_transform_values(
            Z_dict, node_perm_dict, shape=[self.embedder.repr_dim], device=self.device
        )
        return Z

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pos_walks, neg_walks, node_meta = batch

        Z_emb = self.forward_emb_flat(node_meta)

        loss = self.n2v.loss(pos_walks, neg_walks, Z_emb)
        return loss


@dataclass
class Het2VecDataModuleParams(GraphDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class Het2VecDataModule(HeteroGraphDataModule):
    hparams: Union[Het2VecDataModuleParams, DataLoaderParams]

    def __init__(self, dataset: GraphDataset, hparams: GraphDataModuleParams, loader_params: DataLoaderParams) -> None:
        hparams.train_on_full_data = True
        super().__init__(dataset, hparams, loader_params)

    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        mapper = ToHeteroMappingTransform(data.num_nodes_dict)

        hdata = data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)
        n2v_sampler = Node2VecSampler(
            hdata.edge_index, hdata.num_nodes,
            hparams=self.hparams.n2v_params,
            transform_meta=mapper.transform
        )

        return n2v_sampler

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return NodesLoader(
            self.train_data.num_nodes,
            transform=self.train_sampler(self.train_data),
            shuffle=True,
            **self.loader_params.to_dict()
        )

    def eval_sampler(self, data: HeteroData) -> Optional[Sampler]:
        return None
