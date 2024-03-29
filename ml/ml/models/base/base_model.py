from typing import Optional, Union, List

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ml.utils import OptimizerParams
from ml.utils.outputs import OutputExtractor


class BaseModel(LightningModule):
    hparams: OptimizerParams

    train_outputs: OutputExtractor = None
    val_outputs: OutputExtractor = None
    test_outputs: OutputExtractor = None

    def __init__(self, optimizer_params: Optional[OptimizerParams] = None) -> None:
        super().__init__()
        if optimizer_params is not None:
            self.save_hyperparameters(optimizer_params.to_dict())
            self.lr = self.hparams.lr
        else:
            self.lr = 0.001

    def on_train_epoch_start(self) -> None:
        self.train_outputs = None

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.train_outputs = OutputExtractor(outputs)

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = None

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_outputs = OutputExtractor(outputs)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = None

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_outputs = OutputExtractor(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, mode='min', patience=3, min_lr=1e-6, verbose=True, factor=0.2,
            ),
            'interval': 'epoch',
            'monitor': 'epoch_loss',
        }

        return [optimizer], [scheduler]
