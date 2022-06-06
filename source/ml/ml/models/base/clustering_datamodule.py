from functools import lru_cache
from typing import Optional, Dict

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from datasets import GraphDataset
from datasets.utils.labels import LabelDict
from ml.data.loaders.embeddings_loader import EmbeddingsLoader
from ml.utils import DataLoaderParams, dict_catv


class ClusteringDataModule(LightningDataModule):
    dataset: Dataset
    graph_dataset: Optional[GraphDataset]
    loader_params: DataLoaderParams

    def __init__(
            self,
            dataset: Dataset,
            graph_dataset: Optional[GraphDataset],
            loader_params: DataLoaderParams
    ):
        super().__init__()
        self.save_hyperparameters(loader_params.to_dict())
        self.loader_params = loader_params
        self.dataset = dataset
        self.graph_dataset = graph_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return EmbeddingsLoader(
            [self.dataset],
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return EmbeddingsLoader(
            [self.dataset],
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return EmbeddingsLoader(
            [self.dataset],
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return EmbeddingsLoader(
            [self.dataset],
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def _extract_labels(
            self, data: HeteroData
    ) -> Dict[str, LabelDict]:
        node_labels = {}
        for label in self.graph_dataset.labels():
            node_labels[label] = getattr(data, f'{label}_dict')

        return node_labels

    def labels_dict(self) -> Dict[str, LabelDict]:
        if self.graph_dataset is None:
            return {}
        else:
            return self._extract_labels(self.graph_dataset.data)

    @lru_cache(maxsize=1)
    def labels(self) -> Dict[str, Tensor]:
        return {
            label_name: dict_catv(labels)
            for label_name, labels in self.labels_dict().items()
        }
