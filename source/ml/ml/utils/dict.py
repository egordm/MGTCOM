from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union, Dict, Callable, Any

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor


def merge_dicts(ds, merge_fn=None):
    res = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            res[k].append(v)

    if merge_fn is not None:
        for k, v in res.items():
            res[k] = merge_fn(v)

    return res


def dicts_extract(ds, key):
    return [d[key] for d in ds if key in d]


def flat_iter(l):
    for el in l:
        if isinstance(el, list):
            yield from flat_iter(el)
        else:
            yield el


@dataclass
class OutputExtractor:
    outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]

    def extract(self, key) -> List[Tensor]:
        return dicts_extract(flat_iter(self.outputs), key)

    def extract_cat(self, key) -> Union[Tensor, float]:
        return torch.cat(self.extract(key), dim=0)

    def extract_cat_dict(self, key) -> Dict[str, Tensor]:
        return merge_dicts(self.extract(key), lambda xs: torch.cat(xs, dim=0))

    def extract_mean(self, key) -> Union[Tensor, float]:
        return sum(self.extract(key)) / len(self.outputs)


def values_apply(d, fn: Callable[[Any], Any]):
    return {k: fn(v) for k, v in d.items()}