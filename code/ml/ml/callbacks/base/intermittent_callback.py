from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.trainer.states import RunningStage

from ml.utils import HParams


@dataclass
class IntermittentCallbackParams(HParams):
    interval: int = 1
    """Interval between callbacks."""


T = TypeVar("T", bound=IntermittentCallbackParams)


class IntermittentCallback(Callback, Generic[T]):
    hparams: T

    def __init__(self, hparams: T) -> None:
        super().__init__()
        self.hparams = hparams

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        if trainer.state.stage != RunningStage.VALIDATING and trainer.state.stage != RunningStage.TRAINING:
            return

        if (trainer.current_epoch + 1) % self.hparams.interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        self.on_validation_epoch_end_run(trainer, pl_module)

    def validation_can_run(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return (trainer.current_epoch + 1) % self.hparams.interval != 0 \
               and trainer.current_epoch != trainer.max_epochs

    @abstractmethod
    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_end(trainer, pl_module)

        if trainer.state.stage != RunningStage.TESTING:
            return

        self.on_test_epoch_end_run(trainer, pl_module)

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass
