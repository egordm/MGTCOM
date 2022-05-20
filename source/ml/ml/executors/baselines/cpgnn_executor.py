from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs, T
from ml.models.cpgnn import CPGNNDataModuleParams, CPGNNModelParams, CPGNNDataModule, CPGNNModel
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMTopoDataModuleParams, MGCOMFeatModel, MGCOMTopoDataModule, \
    MGCOMFeatTopoModel
from ml.models.node2vec import UnsupervisedLoss
from ml.utils import dataset_choices, Metric


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: CPGNNModelParams = CPGNNModelParams()
    data_params: CPGNNDataModuleParams = CPGNNDataModuleParams()


class MGCOMTopoExecutor(BaseExecutor[MGCOMFeatTopoModel]):
    args: Args
    datamodule: MGCOMTopoDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        self.args.hparams.k_length = self.args.data_params.sampler_params.k_length
        self.args.data_params.num_layers_aux = self.args.hparams.num_layers_aux
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()

        return CPGNNDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        if self.args.hparams.metric != Metric.DOTP:
            self.logger.error(f'Metric {self.args.hparams.metric} not supported for this task. Don\'t use it.')

        return cls(
            metadata=self.datamodule.metadata,
            num_nodes_dict=self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    @property
    def model_cls(self) -> Type[MGCOMFeatTopoModel]:
        return CPGNNModel

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
    MGCOMTopoExecutor().cli()
