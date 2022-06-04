from dataclasses import dataclass
from typing import Type, List

import torch
from pytorch_lightning import Callback, LightningDataModule

from datasets import GraphDataset
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.data.transforms.compute_degree import compute_degree
from ml.executors.base import BaseExecutor, BaseExecutorArgs
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMTopoDataModuleParams, MGCOMTopoDataModule, \
    MGCOMFeatTopoModel
from ml.utils import dataset_choices


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for training."""
    hparams: MGCOMFeatModelParams = MGCOMFeatModelParams()
    data_params: MGCOMTopoDataModuleParams = MGCOMTopoDataModuleParams()


class MGCOMTopoExecutor(BaseExecutor[MGCOMFeatTopoModel]):
    args: Args
    datamodule: MGCOMTopoDataModule

    TASK_NAME = 'embedding_topo'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        dataset: GraphDataset = DATASET_REGISTRY[self.args.dataset]()
        return MGCOMTopoDataModule(
            dataset=dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        embed_mask_dict = None
        # Create embedding mask (used for ablations to check the embedding ratio needed)
        if len(self.args.hparams.embed_node_types) > 0 and self.args.hparams.embed_node_ratio < 1:
            degree_dict = {
                node_type: degrees
                for node_type, degrees in compute_degree(self.datamodule.train_data).items()
                if node_type in self.args.hparams.embed_node_types
            }
            degrees = torch.cat(list(degree_dict.values()), dim=0).sort(descending=True)[0]
            q = degrees[int(len(degrees) * self.args.hparams.embed_node_ratio)]
            embed_mask_dict = {
                node_type: (degrees >= q)
                for node_type, degrees in degree_dict.items()
            }
            self.logger.info(f'Real embedding ratio: {int((degrees >= q).sum())}/{len(degrees)}')

        return cls(
            metadata=self.datamodule.metadata,
            num_nodes_dict=self.datamodule.num_nodes_dict,
            hparams=self.args.hparams,
            optimizer_params=self.args.optimizer_params,
            embed_mask_dict=embed_mask_dict,
        )

    @property
    def model_cls(self) -> Type[MGCOMFeatTopoModel]:
        return MGCOMFeatTopoModel

    def _callbacks(self) -> List[Callback]:
        return [
            *self._embedding_task_callbacks(),
            EmbeddingVisualizerCallback(
                self.datamodule,
                hparams=self.args.callback_params.embedding_visualizer
            ),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
        ]

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMTopoExecutor().cli()
