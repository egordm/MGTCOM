from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import ProgressBar


class CustomProgressBar(ProgressBar):
    def on_train_epoch_start(self, trainer: Trainer, *_: Any) -> None:
        if trainer.current_epoch:
            print()
