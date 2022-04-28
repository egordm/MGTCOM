from typing import Any, Dict, Union

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self) -> None:
        super().__init__(refresh_rate=1, process_position=0)

    # noinspection PyMethodOverriding
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch:
            print()

        super().on_train_epoch_start(trainer, pl_module)
        pl_module.log('epoch', trainer.current_epoch, on_epoch=True)
