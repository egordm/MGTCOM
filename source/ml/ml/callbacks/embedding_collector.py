from typing import Optional, Any, Dict

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ml.utils.dict import merge_dicts


class EmbeddingsCollectorCallback(Callback):
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.emb_batches = []

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if dataloader_idx == 0:
            self.emb_batches.append(outputs['emb'])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        pl_module.val_emb_dict = merge_dicts(self.emb_batches, lambda xs: torch.cat(xs, dim=0))

