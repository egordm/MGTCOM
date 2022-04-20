from enum import Enum
from functools import reduce
from typing import Dict, Union, List, Optional, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torch_geometric.typing import NodeType

from ml.utils import OutputExtractor, OptimizerParams, values_apply


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

    train_outputs: Dict[str, Any] = None
    val_outputs: Dict[str, Any] = None
    test_outputs: Dict[str, Any] = None

    def __init__(self, optimizer_params: Optional[OptimizerParams] = None) -> None:
        super().__init__()
        if optimizer_params is not None:
            self.save_hyperparameters(optimizer_params.to_dict())

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {}

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class EmbeddingModel(BaseModel):
    @property
    def repr_dim(self):
        raise NotImplementedError()

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z=self.forward(batch).detach().cpu(),
            batch_idx=batch_idx,
        )

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.val_outputs['Z'] = outputs.extract_cat('Z')

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z=self.forward(batch).detach().cpu(),
            batch_idx=batch_idx,
        )

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.test_outputs['Z'] = outputs.extract_cat('Z')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return dict(
            Z=self.forward(batch),
            batch_idx=batch_idx,
        )


class HeteroEmbeddingModel(BaseModel):
    @property
    def repr_dim(self):
        raise NotImplementedError()

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=values_apply(self.forward(batch), lambda x: x.detach().cpu()),
            batch_idx=batch_idx,
        )

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.val_outputs['Z_dict'] = outputs.extract_cat_dict('Z_dict')

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=values_apply(self.forward(batch), lambda x: x.detach().cpu()),
            batch_idx=batch_idx,
        )

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.test_outputs['Z_dict'] = outputs.extract_cat_dict('Z_dict')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return dict(
            Z_dict=values_apply(self.forward(batch), lambda x: x.detach().cpu()),
            batch_idx=batch_idx,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # optimizer = torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        return optimizer
