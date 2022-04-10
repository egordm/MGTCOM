from enum import Enum
from typing import Dict

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch_geometric.typing import NodeType

from datasets import GraphDataset
from ml.utils import merge_dicts, dicts_extract


class ValidationDataloaderIdx(Enum):
    SequentialNodes = 0


class BaseEmbeddingModel(pl.LightningModule):
    val_embs: Dict[NodeType, Tensor]
    dataset: GraphDataset

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataloader_idx = ValidationDataloaderIdx(dataloader_idx)

        if dataloader_idx == ValidationDataloaderIdx.SequentialNodes:
            return {
                'emb': self.forward(batch),
            }
        else:
            raise NotImplementedError

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        dataloader_idx = ValidationDataloaderIdx(dataloader_idx)

        if dataloader_idx == ValidationDataloaderIdx.SequentialNodes:
            self.val_embs = merge_dicts(dicts_extract(outputs, 'emb'), lambda xs: torch.cat(xs, dim=0))
