from typing import Union, Sized, Callable, List

import torch
from torch.utils.data import Dataset

from ml.data.loaders.base import BatchedLoader


class EmbeddingsLoader(BatchedLoader):
    def __init__(
            self,
            datasets: List[Union[Sized, Dataset]],
            transform: Callable = None,
            batch_size_tmp: int = None,
            *args, **kwargs
    ):
        kwargs.pop('dataset', None)
        size = len(next(iter(datasets)))
        super().__init__(torch.arange(size), transform, *args, **kwargs, batch_size_tmp=batch_size_tmp)
        self.datasets = datasets
        self.size = size
        assert all(len(d) == self.size for d in self.datasets), "All datasets must have the same size"

    def sample(self, idx):
        return torch.cat([d[idx] for d in self.datasets], dim=1)
