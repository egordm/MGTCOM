from dataclasses import dataclass
from typing import Type, Tuple

from pytorch_lightning import LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutorArgs
from ml.executors.mgcom_topo_executor import MGCOMTopoExecutor
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMFeatTempoModel
from ml.models.mgcom_feat import MGCOMTempoDataModuleParams, MGCOMTempoDataModule
from ml.models.node2vec import UnsupervisedLoss
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: MGCOMFeatModelParams = MGCOMFeatModelParams()
    data_params: MGCOMTempoDataModuleParams = MGCOMTempoDataModuleParams()


class MGCOMTempoExecutor(MGCOMTopoExecutor):
    args: Args
    datamodule: MGCOMTempoDataModule

    TASK_NAME = 'embedding_tempo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    @property
    def model_cls(self) -> Type[MGCOMFeatTempoModel]:
        return MGCOMFeatTempoModel

    def _datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOMTempoDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def run_name(self):
        return self.args.dataset

    def _metric_monitor(self) -> Tuple[str, str]:
        return 'epoch_loss', 'min'


if __name__ == '__main__':
    MGCOMTempoExecutor().cli()
