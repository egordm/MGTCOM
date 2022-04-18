from typing import Union, Sized, Callable, List

import torch
from torch.utils.data import Dataset

from ml.data.loaders.base import BatchedLoader


class EmbeddingsLoader(BatchedLoader):
    def __init__(
            self,
            datasets: List[Union[Sized, Dataset]],
            transform: Callable = None,
            *args, **kwargs
    ):
        self.datasets = datasets
        self.size = len(next(iter(datasets)))
        assert all(len(d) == self.size for d in self.datasets), "All datasets must have the same size"

        super().__init__(torch.arange(self.size), transform, *args, **kwargs)

    def sample(self, idx):
        return torch.cat([d[idx] for d in self.datasets], dim=1)
