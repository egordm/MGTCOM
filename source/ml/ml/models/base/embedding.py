from enum import Enum
from functools import reduce
from typing import Dict, Union, List, Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torch_geometric.typing import NodeType

from ml.utils import OutputExtractor, OptimizerParams


class EmbeddingCombineMode(Enum):
    """
    Defines the way the embeddings are combined.
    """
    ADD = 'add'
    MULT = 'mul'
    CONCAT = 'concat'

    @property
    def combine_fn(self):
        if self == EmbeddingCombineMode.ADD:
            return lambda xs: reduce(torch.Tensor.add_, xs, torch.zeros_like(xs[0]))
        elif self == EmbeddingCombineMode.MULT:
            return lambda xs: reduce(torch.Tensor.mul_, xs, torch.ones_like(xs[0]))
        elif self == EmbeddingCombineMode.CONCAT:
            return lambda xs: torch.cat(xs, dim=-1)
        else:
            raise ValueError(f"Unknown combine mode {self}")


class BaseEmbeddingModel(pl.LightningModule):
    hparams: OptimizerParams
    val_Z_dict: Dict[NodeType, Tensor] = None
    test_Z_dict: Dict[NodeType, Tensor] = None

    @property
    def repr_dim(self):
        raise NotImplementedError()

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.val_Z_dict = outputs.extract_cat_dict('Z_dict')
        # self.val_Z = torch.cat(list(self.val_Z_dict.values()), dim=0)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.test_Z_dict = outputs.extract_cat_dict('Z_dict')
        # self.test_Z = torch.cat(list(self.test_Z_dict.values()), dim=0)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return dict(
            Z_dict=self.forward(batch)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
