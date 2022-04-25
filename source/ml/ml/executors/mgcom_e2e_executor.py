from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.mgcom_combi import MGCOMCombiModelParams, MGCOMCombiDataModuleParams, MGCOMCombiDataModule, \
    MGCOMCombiModel
from ml.models.mgcom_e2e import MGCOME2EDataModule, MGCOME2EModel, MGCOME2EModelParams
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    hparams: MGCOME2EModelParams = MGCOME2EModelParams()
    data_params: MGCOMCombiDataModuleParams = MGCOMCombiDataModuleParams()


class MGCOME2EExecutor(BaseExecutor):
    args: Args
    datamodule: MGCOMCombiDataModule

    TASK_NAME = 'embedding_combi'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOME2EDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        self.args.hparams.combi_params.use_topo = self.args.data_params.use_topo
        self.args.hparams.combi_params.use_tempo = self.args.data_params.use_tempo

        return MGCOME2EModel(
            self.datamodule.metadata, self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return [
            *self._embedding_task_callbacks(),
            ClusteringVisualizerCallback(
                hparams=self.args.callback_params.clustering_visualizer
            ),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOME2EExecutor().cli()
