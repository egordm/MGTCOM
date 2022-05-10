from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.mgcom_combi import MGCOMCombiDataModuleParams, MGCOMCombiDataModule
from ml.models.mgcom_e2e import MGCOME2EDataModule, MGCOME2EModel, MGCOME2EModelParams
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: MGCOME2EModelParams = MGCOME2EModelParams()
    data_params: MGCOMCombiDataModuleParams = MGCOMCombiDataModuleParams()


class MGCOME2EExecutor(BaseExecutor[MGCOME2EModel]):
    args: Args
    datamodule: MGCOMCombiDataModule

    TASK_NAME = 'embedding_combi'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOME2EDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        self.args.hparams.combi_params.use_topo = self.args.data_params.use_topo
        self.args.hparams.combi_params.use_tempo = self.args.data_params.use_tempo

        return cls(
            metadata=self.datamodule.metadata,
            num_nodes_dict=self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    @property
    def model_cls(self) -> Type[MGCOME2EModel]:
        return MGCOME2EModel

    def _callbacks(self) -> List[Callback]:
        return [
            *self._embedding_task_callbacks(),
            ClusteringVisualizerCallback(
                hparams=self.args.callback_params.clustering_visualizer
            ),
            ClusteringEvalCallback(
                self._datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOME2EExecutor().cli()
