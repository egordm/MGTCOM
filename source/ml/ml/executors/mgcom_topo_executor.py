from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMTopoDataModuleParams, MGCOMFeatModel, MGCOMTopoDataModule, \
    MGCOMFeatTopoModel
from ml.models.node2vec import UnsupervisedLoss
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: MGCOMFeatModelParams = MGCOMFeatModelParams()
    data_params: MGCOMTopoDataModuleParams = MGCOMTopoDataModuleParams()


class MGCOMTopoExecutor(BaseExecutor):
    args: Args
    datamodule: MGCOMTopoDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOMTopoDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        return MGCOMFeatTopoModel(
            self.datamodule.metadata, self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return self._embedding_task_callbacks()

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMTopoExecutor().cli()
