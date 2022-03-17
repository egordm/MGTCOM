import sys
from typing import Union, List

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
