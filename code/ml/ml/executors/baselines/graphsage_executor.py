from dataclasses import dataclass
from typing import Type, List, Tuple

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.graphsage import GraphSAGEDataModuleParams, GraphSAGEDataModule, \
    GraphSAGEModel, GraphSAGEModelParams
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMTopoDataModuleParams, MGCOMTopoDataModule, \
    MGCOMFeatTopoModel
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: GraphSAGEModelParams = GraphSAGEModelParams()
    data_params: GraphSAGEDataModuleParams = GraphSAGEDataModuleParams()


class GraphSAGEExecutor(BaseExecutor[MGCOMFeatTopoModel]):
    args: Args
    datamodule: GraphSAGEDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return GraphSAGEDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        return cls(
            metadata=self.datamodule.metadata,
            num_nodes_dict=self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    @property
    def model_cls(self) -> Type[MGCOMFeatTopoModel]:
        return GraphSAGEModel

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

    def _metric_monitor(self) -> Tuple[str, str]:
        return 'epoch_loss', 'min'


if __name__ == '__main__':
    GraphSAGEExecutor().cli()
