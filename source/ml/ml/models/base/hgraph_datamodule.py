from abc import abstractmethod
from typing import Optional

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from ml.models.base.graph_datamodule import GraphDataModule
from ml.data.loaders.nodes_loader import NodesLoader, HeteroNodesLoader
from ml.data.samplers.base import Sampler


class HomogenousGraphDataModule(GraphDataModule):
    heterogenous: bool = False

    @abstractmethod
    def train_sampler(self) -> Optional[Sampler]:
        raise NotImplementedError()

    @abstractmethod
    def eval_sampler(self) -> Optional[Sampler]:
        raise NotImplementedError()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return NodesLoader(
            self.train_data.num_nodes,
            transform=self.train_sampler(),
            shuffle=True,
            **self.loader_params.to_dict()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return NodesLoader(
            self.val_data.num_nodes,
            transform=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return NodesLoader(
            self.test_data.num_nodes,
            transform=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return NodesLoader(
            self.data.num_nodes,
            transform=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )


class HeterogenousGraphDataModule(GraphDataModule):
    heterogenous: bool = True

    @abstractmethod
    def train_sampler(self) -> Optional[Sampler]:
        raise NotImplementedError()

    @abstractmethod
    def eval_sampler(self) -> Optional[Sampler]:
        raise NotImplementedError()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return HeteroNodesLoader(
            self.train_data.num_nodes_dict,
            transform=self.train_sampler(),
            shuffle=True,
            **self.loader_params.to_dict()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.val_data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.test_data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return HeteroNodesLoader(
            self.data.num_nodes_dict,
            transform_nodes_fn=self.eval_sampler(),
            shuffle=False,
            **self.loader_params.to_dict(),
        )
