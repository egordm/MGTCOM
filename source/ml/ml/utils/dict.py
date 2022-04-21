from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
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


def values_apply(d, fn: Callable[[Any], Any]):
    return {k: fn(v) for k, v in d.items()}


def dict_mapv(d: Dict[str, Any], fn: Callable[[Any], Any]):
    return {k: fn(v) for k, v in d.items()}
