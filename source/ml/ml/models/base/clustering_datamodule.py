from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset

from datasets import GraphDataset
from ml.data.loaders.embeddings_loader import EmbeddingsLoader
from ml.utils import DataLoaderParams


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
            shuffle=True,
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
