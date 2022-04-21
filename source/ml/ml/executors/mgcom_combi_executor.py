from dataclasses import dataclass
from typing import Type, List

from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.embedding_eval_callback import EmbeddingEvalCallback
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.mgcom_combi import MGCOMCombiModelParams, MGCOMCombiDataModuleParams, MGCOMCombiDataModule, \
    MGCOMCombiModel
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    hparams: MGCOMCombiModelParams = MGCOMCombiModelParams()
    data_params: MGCOMCombiDataModuleParams = MGCOMCombiDataModuleParams()


class MGCOMCombiExecutor(BaseExecutor):
    args: Args
    datamodule: MGCOMCombiDataModule

    TASK_NAME = 'embedding_combi'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOMCombiDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        return MGCOMCombiModel(
            self.datamodule.metadata, self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return self._embedding_task_callbacks()

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMCombiExecutor().cli()
