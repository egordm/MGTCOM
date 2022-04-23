from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, List

from pytorch_lightning import LightningDataModule, Callback
from torch.utils.data import Dataset

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.data import PretrainedEmbeddingsDataset, SyntheticGMMDataset
from ml.executors.base import BaseExecutorArgs, BaseExecutor
from ml.models.mgcom_comdet import MGCOMComDetDataModuleParams, MGCOMComDetDataModule, MGCOMComDetModel, \
    MGCOMComDetModelParams
from ml.utils import dataset_choices, DataLoaderParams, TrainerParams
from shared import get_logger

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'community_detection'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    pretrained_path: Optional[str] = None
    hparams: MGCOMComDetModelParams = MGCOMComDetModelParams()
    data_params: MGCOMComDetDataModuleParams = MGCOMComDetDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams(batch_size=200)
    trainer_params: TrainerParams = TrainerParams(max_epochs=200)


class MGCOMComDetExecutor(BaseExecutor):
    args: Args
    datamodule: MGCOMComDetDataModule
    dataset: Dataset

    TASK_NAME = 'community_detection'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def datamodule(self) -> LightningDataModule:
        graph_dataset: Optional[GraphDataset] = DATASET_REGISTRY[self.args.dataset]()

        if self.args.pretrained_path:
            logger.info(f'Using pretrained embeddings from {self.args.pretrained_path}')
            dataset = PretrainedEmbeddingsDataset.from_pretrained(self.args.pretrained_path, graph_dataset.name)
        else:
            logger.info('No pretrained embeddings provided, using synthetic dataset')
            dataset = SyntheticGMMDataset()
            graph_dataset = None

        self.dataset = dataset

        return MGCOMComDetDataModule(
            dataset=dataset,
            graph_dataset=graph_dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model(self):
        return MGCOMComDetModel(
            self.dataset.repr_dim,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
        )

    def callbacks(self) -> List[Callback]:
        return [
            ClusteringVisualizerCallback(hparams=self.args.callback_params.clustering_visualizer),
            ClusteringEvalCallback(self.datamodule, hparams=self.args.callback_params.clustering_eval),
            SaveEmbeddingsCallback(),
            SaveGraphCallback(
                self.datamodule.graph_dataset.data,
                node_labels={},
                hparams=self.args.callback_params.save_graph,
                clustering=True,
            ) if self.datamodule.graph_dataset is not None else None,
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMComDetExecutor().cli()
