from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Callable, Union

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor

from ml.utils import flat_iter, dicts_extract, merge_dicts


class ExtractMode(Enum):
    CAT = 0
    CAT_DICT = 1
    MEAN = 2


@dataclass
class OutputExtractor:
    outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    outputs_cache: Dict[str, Any] = field(default_factory=dict)

    def extract(self, key, cache=False) -> List[Tensor]:
        return self.check_cache(key, lambda: dicts_extract(flat_iter(self.outputs), key), cache)

    def extract_cat(self, key, cache=False) -> Tensor:
        return self.check_cache(key, lambda: torch.cat(self.extract(key), dim=0), cache)

    def extract_cat_dict(self, key, cache=False) -> Dict[str, Tensor]:
        return self.check_cache(key, lambda: merge_dicts(self.extract(key), lambda xs: torch.cat(xs, dim=0)), cache)

    def extract_cat_kv(self, key, cache=False) -> Tensor:
        return self.check_cache(key, lambda: torch.cat(list(self.extract_cat_dict(key, False).values()), dim=0), cache)

    def extract_mean(self, key, cache=False) -> Union[Tensor, float]:
        return self.check_cache(key, lambda: sum(self.extract(key)) / len(self.outputs), cache)

    def __contains__(self, key):
        return any(key in d for d in flat_iter(self.outputs))

    def extract_item(self, key, mode: ExtractMode, cache=False) -> Any:
        if mode == ExtractMode.CAT:
            output = self.extract_cat(key, cache)
        elif mode == ExtractMode.CAT_DICT:
            output = self.extract_cat_dict(key, cache)
        elif mode == ExtractMode.MEAN:
            output = self.extract_mean(key, cache)
        else:
            raise ValueError(f"Unknown extract mode: {mode}")

        return output

    def check_cache(self, key, miss_fn: Callable[[], Any], cache=True) -> Any:
        if key not in self.outputs_cache:
            if not cache:
                return miss_fn()

            self.outputs_cache[key] = miss_fn()

        return self.outputs_cache[key]
