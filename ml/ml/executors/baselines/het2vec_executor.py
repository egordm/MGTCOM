from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs, T
from ml.layers.embedding import HeteroNodeEmbedding
from ml.models.het2vec import Het2VecModel, Het2VecDataModule, Het2VecDataModuleParams, Het2VecClusModelParams, \
    Het2VecClusModel
from ml.models.mgcom_feat import MGCOMTopoDataModule
from ml.utils import dataset_choices, Metric, OptimizerParams


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: Het2VecClusModelParams = Het2VecClusModelParams()
    optimizer_params = OptimizerParams()
    data_params: Het2VecDataModuleParams = Het2VecDataModuleParams()


class Het2VecExecutor(BaseExecutor[Het2VecModel]):
    args: Args
    datamodule: MGCOMTopoDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return Het2VecDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        embedder = HeteroNodeEmbedding(
            self.datamodule.data.num_nodes_dict,
            self.args.hparams.repr_dim,
        )

        return cls(
            embedder=embedder,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    @property
    def model_cls(self) -> Type[Het2VecModel]:
        return Het2VecClusModel

    def _callbacks(self) -> List[Callback]:
        return [
            *self._embedding_task_callbacks(),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    Het2VecExecutor().cli()
