from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from typing import Dict, Union, List, Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from ml.utils import OptimizerParams, values_apply, dict_mapv
from ml.utils.outputs import OutputExtractor


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


class BaseModel(pl.LightningModule):
    hparams: OptimizerParams

    train_outputs: OutputExtractor = None
    val_outputs: OutputExtractor = None
    test_outputs: OutputExtractor = None

    def __init__(self, optimizer_params: Optional[OptimizerParams] = None) -> None:
        super().__init__()
        if optimizer_params is not None:
            self.save_hyperparameters(optimizer_params.to_dict())

        self.lr = self.hparams.lr

    def on_train_epoch_start(self) -> None:
        self.train_outputs = {}

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.train_outputs = OutputExtractor(outputs)

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.val_outputs = OutputExtractor(outputs)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {}

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.test_outputs = OutputExtractor(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class BaseEmbeddingModel(BaseModel):
    heterogeneous: bool = False

    @property
    def repr_dim(self):
        raise NotImplementedError()


class EmbeddingModel(BaseEmbeddingModel, ABC):
    heterogeneous: bool = False

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().train_epoch_end(outputs)
        self.log('epoch_loss', self.train_outputs.extract_mean('loss'), prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}


class HeteroEmbeddingModel(BaseEmbeddingModel, ABC):
    heterogeneous: bool = True

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}
