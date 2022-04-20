from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.lp_eval_callback import LPEvalCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.layers.embedding import NodeEmbedding
from ml.models.node2vec import Node2VecDataModule, Node2VecDataModuleParams, Node2VecModel
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    metric: Metric = Metric.DOTP
    repr_dim: int = 128
    optimizer_params = OptimizerParams()
    data_params: Node2VecDataModuleParams = Node2VecDataModuleParams()


class Node2VecExecutor(BaseExecutor):
    args: Args
    datamodule: Node2VecDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return Node2VecDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        embedder = NodeEmbedding(
            self.datamodule.train_data.num_nodes,
            self.args.repr_dim,
        )
        return Node2VecModel(
            embedder=embedder,
            metric=self.args.metric,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return [
            LPEvalCallback(self.datamodule)
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    Node2VecExecutor().cli()
