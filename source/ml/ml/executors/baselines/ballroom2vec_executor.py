from dataclasses import dataclass
from typing import Type

from pytorch_lightning import LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutorArgs
from ml.executors.baselines.node2vec_executor import Node2VecExecutor
from ml.layers.embedding import NodeEmbedding
from ml.models.ballroom2vec import Ballroom2VecDataModuleParams, Ballroom2VecDataModule, Ballroom2VecModel
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    metric: Metric = Metric.DOTP
    repr_dim: int = 128
    optimizer_params = OptimizerParams()
    data_params: Ballroom2VecDataModuleParams = Ballroom2VecDataModuleParams()


class Ballroom2VecExecutor(Node2VecExecutor):
    args: Args
    datamodule: Ballroom2VecDataModule

    TASK_NAME = 'embedding_tempo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def model(self):
        embedder = NodeEmbedding(
            self.datamodule.train_data.num_nodes,
            self.args.repr_dim,
        )
        return Ballroom2VecModel(
            embedder=embedder,
            metric=self.args.metric,
            optimizer_params=self.args.optimizer_params,
        )

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return Ballroom2VecDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )


if __name__ == '__main__':
    Ballroom2VecExecutor().cli()
