from dataclasses import dataclass
from typing import Type

from pytorch_lightning import LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutorArgs
from ml.executors.baselines.node2vec_executor import Node2VecExecutor
from ml.layers.embedding import NodeEmbedding
from ml.models.ctdne import CTDNEDataModule, CTDNEModel, CTDNEDataModuleParams
from ml.models.node2vec import UnsupervisedLoss, Node2VecWrapperModelParams, Node2VecModel
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: Node2VecWrapperModelParams = Node2VecWrapperModelParams(loss=UnsupervisedLoss.SKIPGRAM, metric=Metric.DOTP)
    optimizer_params = OptimizerParams()
    data_params: CTDNEDataModuleParams = CTDNEDataModuleParams()


class CTDNEExecutor(Node2VecExecutor):
    args: Args
    datamodule: CTDNEDataModule

    TASK_NAME = 'embedding_tempo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    @property
    def model_cls(self) -> Type[Node2VecModel]:
        return CTDNEModel

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return CTDNEDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )


if __name__ == '__main__':
    CTDNEExecutor().cli()
