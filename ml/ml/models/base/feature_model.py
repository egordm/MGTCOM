from abc import ABC
from enum import Enum
from functools import reduce
from typing import Union, List, Optional, Any

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from ml.models.base.base_model import BaseModel
from ml.utils import dict_mapv


class FeatureCombineMode(Enum):
    """
    Defines the way the embeddings are combined.
    """
    ADD = 'add'
    MULT = 'mul'
    CONCAT = 'concat'

    @property
    def combine_fn(self):
        if self == FeatureCombineMode.ADD:
            return lambda xs: reduce(torch.Tensor.add_, xs, torch.zeros_like(xs[0]))
        elif self == FeatureCombineMode.MULT:
            return lambda xs: reduce(torch.Tensor.mul_, xs, torch.ones_like(xs[0]))
        elif self == FeatureCombineMode.CONCAT:
            return lambda xs: torch.cat(xs, dim=-1)
        else:
            raise ValueError(f"Unknown combine mode {self}")


class BaseFeatureModel(BaseModel):
    heterogeneous: bool = False

    @property
    def repr_dim(self):
        raise NotImplementedError()


class FeatureModel(BaseFeatureModel, ABC):
    heterogeneous: bool = False

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)
        self.log('epoch_loss', self.train_outputs.extract_mean('loss'), prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {'Z': self.forward(batch).detach().cpu(), 'batch_idx': batch_idx}


class HeteroFeatureModel(BaseFeatureModel, ABC):
    heterogeneous: bool = True

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)
        self.log('epoch_loss', self.train_outputs.extract_mean('loss'), prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {'Z_dict': dict_mapv(self.forward(batch), lambda x: x.detach().cpu()), 'batch_idx': batch_idx}
