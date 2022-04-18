from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress import ProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self) -> None:
        super().__init__(refresh_rate=1, process_position=0)

    def on_train_epoch_start(self, trainer: Trainer, *_: Any) -> None:
        if trainer.current_epoch:
            print()

        super().on_train_epoch_start(trainer, *_)


