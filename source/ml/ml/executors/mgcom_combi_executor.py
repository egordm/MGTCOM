from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.executors.mgcom_topo_executor import MGCOMTopoExecutor
from ml.models.mgcom_combi import MGCOMCombiModelParams, MGCOMCombiDataModuleParams, MGCOMCombiDataModule, \
    MGCOMCombiModel
from ml.models.node2vec import UnsupervisedLoss
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: MGCOMCombiModelParams = MGCOMCombiModelParams(use_cluster=False)
    data_params: MGCOMCombiDataModuleParams = MGCOMCombiDataModuleParams()


class MGCOMCombiExecutor(MGCOMTopoExecutor):
    args: Args
    datamodule: MGCOMCombiDataModule

    TASK_NAME = 'embedding_combi'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    @property
    def model_cls(self) -> Type[MGCOMCombiModel]:
        return MGCOMCombiModel

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOMCombiDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )


if __name__ == '__main__':
    MGCOMCombiExecutor().cli()
