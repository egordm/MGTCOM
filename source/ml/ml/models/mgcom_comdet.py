from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset

from ml.data.loaders.embeddings_loader import EmbeddingsLoader
from ml.utils import HParams, DataLoaderParams


@dataclass
class MGCOMComDetDataModuleParams(HParams):
    pass


class MGCOMComDetDataModule(pl.LightningDataModule):
    dataset: Dataset

    def __init__(
            self,
            dataset: Dataset,
            hparams: MGCOMComDetDataModuleParams,
            loader_params: DataLoaderParams
    ):
        self.save_hyperparameters(hparams.to_dict())
        self.save_hyperparameters(loader_params.to_dict())
        self.loader_params = loader_params

        self.dataset = dataset
        super().__init__()

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


