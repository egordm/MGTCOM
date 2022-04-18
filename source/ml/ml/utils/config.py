from dataclasses import dataclass
from typing import Optional, Type, Dict

import pytorch_lightning as pl
from simple_parsing import Serializable, choice

from datasets.utils.base import DATASET_REGISTRY


class HParams(Serializable):
    def to_dict(self, dict_factory: Type[Dict] = pl.utilities.AttributeDict, recurse: bool = True) -> Dict:
        return super().to_dict(dict_factory, recurse)


@dataclass
class TrainerParams(HParams):
    max_epochs: int = 20
    """Number of epochs to train for."""
    gpus: Optional[int] = None
    """GPUs to use. If None, use CPU."""


@dataclass
class OptimizerParams(HParams):
    lr: float = 0.01
    """Learning rate"""


@dataclass
class DataLoaderParams(HParams):
    num_workers: int = 0
    """Number of workers to use for data loading"""
    batch_size: int = 16
    """Batch size for training, validation and test"""


def dataset_choices():
    return choice(*DATASET_REGISTRY.names, default="StarWars")
