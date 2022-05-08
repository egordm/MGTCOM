from dataclasses import dataclass
from typing import Type

from pytorch_lightning import LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutorArgs
from ml.executors.baselines.node2vec_executor import Node2VecExecutor
from ml.layers.embedding import NodeEmbedding
from ml.models.ballroom2vec import Ballroom2VecDataModuleParams, Ballroom2VecDataModule, Ballroom2VecModel
from ml.models.node2vec import UnsupervisedLoss, Node2VecWrapperModelParams, Node2VecModel
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: Node2VecWrapperModelParams = Node2VecWrapperModelParams()
    optimizer_params = OptimizerParams()
    data_params: Ballroom2VecDataModuleParams = Ballroom2VecDataModuleParams()


class Ballroom2VecExecutor(Node2VecExecutor):
    args: Args
    datamodule: Ballroom2VecDataModule

    TASK_NAME = 'embedding_tempo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    @property
    def model_cls(self) -> Type[Node2VecModel]:
        return Ballroom2VecModel

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return Ballroom2VecDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )


if __name__ == '__main__':
    Ballroom2VecExecutor().cli()
