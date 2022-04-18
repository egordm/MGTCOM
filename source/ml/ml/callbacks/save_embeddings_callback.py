from pathlib import Path
from typing import List, Any

import torch
from pytorch_lightning import Callback, Trainer, LightningModule

from ml.utils import OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveEmbeddingsCallback(Callback):
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: List[Any]) -> None:
        outputs = OutputExtractor(outputs)
        Z_dict = outputs.extract_cat_dict('Z_dict')

        for train_logger in trainer.loggers:
            if train_logger.save_dir is not None:
                save_dir = Path(train_logger.save_dir)
                logger.info(f'Saving embeddings for {train_logger.name} in: {save_dir}')
                torch.save(Z_dict, save_dir / 'embeddings.pt')
