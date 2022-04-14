from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union

import torch
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
    outputs: dict

    def extract(self, key) -> List[Tensor]:
        return dicts_extract(flat_iter(self.outputs), key)

    def extract_cat(self, key) -> Union[Tensor, float]:
        return torch.cat(self.extract(key), dim=0)

    def extract_mean(self, key) -> Union[Tensor, float]:
        return sum(self.extract(key)) / len(self.outputs)
