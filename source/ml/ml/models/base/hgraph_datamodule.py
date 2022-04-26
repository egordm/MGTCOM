from abc import abstractmethod
from typing import Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.data import HeteroData, Data

from ml.data.loaders.nodes_loader import NodesLoader, HeteroNodesLoader
from ml.data.samplers.base import Sampler
from ml.models.base.graph_datamodule import GraphDataModule


class HomogenousGraphDataModule(GraphDataModule):
    heterogenous: bool = False

    @abstractmethod
    def train_sampler(self, data: Data) -> Optional[Sampler]:
        raise NotImplementedError()

    @abstractmethod
    def eval_sampler(self, data: Data) -> Optional[Sampler]:
        raise NotImplementedError()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return NodesLoader(
            self.train_data.num_nodes,
            transform=self.train_sampler(self.train_data),
            shuffle=True,
            **self.loader_params.to_dict()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return NodesLoader(
            self.val_data.num_nodes,
            transform=self.eval_sampler(self.val_data),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return NodesLoader(
            self.test_data.num_nodes,
            transform=self.eval_sampler(self.test_data),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        # Node order dict is in same order as self.data
        if 'id' in self.test_data.keys:
            node_order = torch.argsort(self.test_data.id)
        else:
            node_order = None

        return NodesLoader(
            self.test_data.num_nodes,
            node_order=node_order,
            transform=self.eval_sampler(self.test_data),
            shuffle=False,
            **self.loader_params.to_dict(),
        )


class HeteroGraphDataModule(GraphDataModule):
    heterogenous: bool = True

    @abstractmethod
    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        raise NotImplementedError()

    @abstractmethod
    def eval_sampler(self, data: HeteroData) -> Optional[Sampler]:
        raise NotImplementedError()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return HeteroNodesLoader(
            self.train_data.num_nodes_dict,
            transform_nodes_fn=self.train_sampler(self.train_data),
            shuffle=True,
            **self.loader_params.to_dict()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.val_data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(self.val_data),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.test_data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(self.test_data),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        # Node order dict is in same order as self.data
        if 'id' in self.test_data.keys:
            node_order_dict = {
                node_type: torch.argsort(ids)
                for node_type, ids in self.test_data.id_dict.items()
            }
        else:
            node_order_dict = None

        return HeteroNodesLoader(
            self.test_data.num_nodes_dict,
            node_order_dict=node_order_dict,
            transform_nodes_fn=self.eval_sampler(self.test_data),  # TODO: embedding methods wont like this!
            shuffle=False,
            **self.loader_params.to_dict(),
        )
