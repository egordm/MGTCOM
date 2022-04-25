from typing import Any, Dict, Union

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self) -> None:
        super().__init__(refresh_rate=1, process_position=0)

    def on_train_epoch_start(self, trainer: Trainer, *_: Any) -> None:
        if trainer.current_epoch:
            print()

        super().on_train_epoch_start(trainer, *_)
        trainer.logger.log_metrics({
            'trainer/epoch': trainer.current_epoch,
        })
