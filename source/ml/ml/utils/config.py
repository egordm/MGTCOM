from dataclasses import dataclass
from typing import Type, Dict

import pytorch_lightning as pl
from simple_parsing import Serializable, choice

from datasets.utils.graph_dataset import DATASET_REGISTRY


class HParams(Serializable):
    def to_dict(self, dict_factory: Type[Dict] = pl.utilities.AttributeDict, recurse: bool = True) -> Dict:
        return super().to_dict(dict_factory, recurse)


@dataclass
class TrainerParams(HParams):
    max_epochs: int = 20
    """Number of epochs to train for."""
    cpu: bool = False
    """Whether to use CPU or GPU."""
    val_check_interval: float = 1.0
    """Interval between validation epochs"""


@dataclass
class OptimizerParams(HParams):
    lr: float = 0.01
    """Learning rate"""


@dataclass
class DataLoaderParams(HParams):
    num_workers: int = 16
    """Number of workers to use for data loading"""
    batch_size: int = 16
    """Batch size for training, validation and test"""
    persistent_workers = True
    """Whether to use persistent workers"""
    pin_memory = True


def dataset_choices():
    return choice(*DATASET_REGISTRY.names, default="StarWars")
