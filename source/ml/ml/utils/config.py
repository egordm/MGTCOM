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
    max_epochs: int = 8
    gpus: Optional[int] = None


@dataclass
class DataLoaderParams(Serializable):
    num_workers: int = 0
    batch_size: int = 16


def dataset_choices():
    return choice(*DATASET_REGISTRY.names, default="StarWars")
