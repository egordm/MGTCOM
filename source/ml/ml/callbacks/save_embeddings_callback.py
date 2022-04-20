from pathlib import Path
from typing import List, Any

import torch
import wandb
from pytorch_lightning import Callback, Trainer, LightningModule

from ml.models.base.embedding import BaseModel
from ml.utils import OutputExtractor
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveEmbeddingsCallback(Callback):
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: BaseModel, outputs: List[Any]) -> None:
        outputs = OutputExtractor(outputs)
        saved = False
        if outputs.has_key('Z_dict'):
            logger.info('Saving heterogenous embeddings...')
            Z_dict = outputs.extract_cat_dict('Z_dict')
            self.save_embeddings(Z_dict, 'embeddings_hetero.pt')
            saved = True

        if outputs.has_key('Z'):
            logger.info('Saving homogeneous embeddings...')
            Z = outputs.extract_cat('Z')
            self.save_embeddings(Z, 'embeddings_homogeneous.pt')
            saved = True

        if not saved:
            logger.warning('No embeddings to save!')

    def save_embeddings(self, Z: Any, name: str) -> None:
        save_dir = Path(wandb.run.dir) / name
        logger.info(f'Saving embeddings in: {save_dir}')
        torch.save(Z, save_dir)
