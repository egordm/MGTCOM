from abc import abstractmethod

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.trainer.states import RunningStage


class IntermittentCallback(Callback):
    def __init__(self, interval: int = 3) -> None:
        super().__init__()
        self.interval = interval

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        self.on_run(trainer, pl_module)

    @abstractmethod
    def on_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass
