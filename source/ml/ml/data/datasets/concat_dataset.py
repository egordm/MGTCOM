from typing import List, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.size = len(next(iter(datasets)))
        assert all(len(d) == self.size for d in self.datasets), "All datasets must have the same size"

    def __len__(self):
        return self.size

    def __getitem__(self, idx: Union[List[int], Tensor]):
        return torch.cat([d[idx] for d in self.datasets], dim=1)
