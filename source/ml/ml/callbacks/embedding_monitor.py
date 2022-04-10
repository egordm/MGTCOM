from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage

from ml.models.base.embedding import BaseEmbeddingModel
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class EmbeddingMonitor(Callback):
    def __init__(self, logging_interval: int = 5) -> None:
        super().__init__()
        self.logging_interval = logging_interval

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseEmbeddingModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch == 0:
            return

        if trainer.current_epoch % self.logging_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        emb_dict = pl_module.val_embs
        name_dict = pl_module.dataset.data.name_dict

        logger.info(f"Saving embeddings")
        df = pd.concat([
            pd.DataFrame({
                'name': name_dict[node_type],
                'node_type': node_type,
                'x': emb_dict[node_type].tolist()
            })
            for node_type in pl_module.dataset.data.node_types
        ])

        wandb_logger: WandbLogger = trainer.logger
        wandb_logger.log_table("embeddings", dataframe=df)

        u = 0
