import sys
from typing import Union

import numpy as np
import torch
from torch import Tensor


def partition_values(vs, ranges):
    partitions = []
    for i, (start, end) in enumerate(ranges):
        size = int(torch.sum((vs >= start) & (vs < end)))
        partitions.append(size)

    partitions = torch.sort(vs).values.split(partitions)

    return partitions


def randint_range(range: Tensor, low=None, dtype=torch.long):
    size = range.size()

    out = torch.randint(0, sys.maxsize, size=size, dtype=dtype) % range
    if low is not None:
        out = out + low
    return out


def ensure_numpy(x: Union[Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def unique_count(I: Tensor, k: int) -> Tensor:
    """
    Counts the number of elements in each of the k partitions.
    """
    labels, counts = torch.unique(I, return_counts=True)
    result = torch.zeros(k, dtype=torch.long)
    result[labels] = counts
    return result
