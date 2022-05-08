from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, List

import torch
from pytorch_lightning import LightningDataModule, Callback
from torch.utils.data import Dataset

from datasets import GraphDataset
from datasets.utils.conversion import igraph_from_hetero
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.algo.dpm import InitMode
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.data import PretrainedEmbeddingsDataset, SyntheticGMMDataset
from ml.executors.base import BaseExecutorArgs, BaseExecutor, T
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
    """Graph Dataset to use for visualization and evaluation."""
    pretrained_path: Optional[str] = None
    hparams: MGCOMComDetModelParams = MGCOMComDetModelParams()
    data_params: MGCOMComDetDataModuleParams = MGCOMComDetDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams(batch_size=200)
    trainer_params: TrainerParams = TrainerParams(max_epochs=200)


class MGCOMComDetExecutor(BaseExecutor[MGCOMComDetModel]):
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

    def model_args(self, cls):
        if self.datamodule.graph_dataset is not None and self.args.hparams.cluster_init_mode == InitMode.HardAssignment:
            logger.info('Initializing clustering model using Louvain labels')
            G, _, _, _ = igraph_from_hetero(self.datamodule.graph_dataset.data)
            com = G.community_multilevel()
            z = torch.tensor(com.membership, dtype=torch.long)
            self.args.hparams.init_k = len(com)
        else:
            z = None

        return cls(
            repr_dim=self.dataset.repr_dim,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
            init_z=z,
        )

    @property
    def model_cls(self) -> Type[MGCOMComDetModel]:
        return MGCOMComDetModel

    def callbacks(self) -> List[Callback]:
        ret = [
            ClusteringVisualizerCallback(
                hparams=self.args.callback_params.clustering_visualizer
            ),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
            SaveEmbeddingsCallback(),
        ]

        if self.datamodule.graph_dataset is not None:
            ret.append(
                SaveGraphCallback(
                    self.datamodule.graph_dataset,
                    hparams=self.args.callback_params.save_graph,
                    clustering=False,
                )
            )

        return ret

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMComDetExecutor().cli()
