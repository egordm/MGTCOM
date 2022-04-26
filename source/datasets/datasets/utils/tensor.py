import sys

import torch
from torch import Tensor


def randint_range(range: Tensor, low=None, dtype=torch.long):
    size = range.size()

    out = torch.randint(0, sys.maxsize, size=size, dtype=dtype) % range
    if low is not None:
        out = out + low
    return out