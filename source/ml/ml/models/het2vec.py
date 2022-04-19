from dataclasses import dataclass
from typing import Union

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor

from datasets import GraphDataset
from ml.algo.transforms import HeteroMappingTransform
from ml.data import Compose
from ml.data.graph_datamodule import GraphDataModuleParams, GraphDataModule
from ml.data.loaders.nodes_loader import HeteroNodesLoader, NodesLoader
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams, Node2VecSampler
from ml.models.node2vec import Node2VecModel
from ml.utils import DataLoaderParams


class Het2VecModel(Node2VecModel):
    def forward(self, batch):
        node_idx_dict, node_perm_dict = batch
        Z_emb = self.embedder(node_idx_dict)

        return Z_emb

    def _forward_emb(self, node_meta) -> Tensor:
        node_idx_dict, node_perm_dict = node_meta
        Z_dict = self.embedder(node_idx_dict)

        # Transform hetero data to homogenous data in the sampled order
        batch_size = sum([len(v) for v in node_idx_dict.values()])
        Z = torch.zeros(batch_size, self.embedder.repr_dim)
        for node_type, node_perm in node_perm_dict.items():
            Z[node_perm] = Z_dict[node_type]

        return Z

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pos_walks, neg_walks, node_meta = batch

        Z_emb = self._forward_emb(node_meta)

        loss = self.loss(pos_walks, neg_walks, Z_emb)
        return loss


@dataclass
class Het2VecDataModuleParams(GraphDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class Het2VecDataModule(GraphDataModule):
    hparams: Union[Het2VecDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: Het2VecDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__(dataset, hparams, loader_params)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data = self.test_data  # Since induction doesnt work on node2vec
        mapper = HeteroMappingTransform(data.num_nodes_dict, to_hetero=True)

        hdata = data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)
        n2v_sampler = Node2VecSampler(hdata.edge_index, hdata.num_nodes, hparams=self.hparams.n2v_params)

        def remap_n2v_batch(batch):
            pos_walks, neg_walks, node_meta = batch
            node_meta_new = mapper(node_meta)
            return (pos_walks, neg_walks, node_meta_new)

        sampler = Compose([
            n2v_sampler,
            remap_n2v_batch
        ])

        return NodesLoader(
            data.num_nodes, transform=sampler,
            shuffle=True,
            **self.loader_params.to_dict(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.val_data.num_nodes_dict,
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.test_data.num_nodes_dict,
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.data.num_nodes_dict,
            shuffle=False,
            **self.loader_params.to_dict(),
        )
