from typing import Union, List

import numpy as np
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.utils.data.dataset import Dataset


class SyntheticGMMDataset(Dataset):
    def __init__(self, n_samples: int = 10000, repr_dim: int = 2, k: int = 15):
        super().__init__()
        self.repr_dim = repr_dim
        self.k = k

        mus = torch.tensor([
            [-5., -5.],
            [-5., -3.],
            [-5., 0.],
            [-5., 3.],
            [-5., 5.],

            [0., -5.],
            [0., -3.],
            [0., 0.],
            [0., 3.],
            [0., 5.],

            [5., -5.],
            [5., -3.],
            [5., 0.],
            [5., 3.],
            [5., 5.],
        ])[:k]

        if mus.shape[1] != self.repr_dim:
            mus = torch.cat([
                mus, torch.rand((k, self.repr_dim - mus.shape[1]))
            ], dim=1)

        covs = [torch.eye(self.repr_dim) * 0.5 + (torch.rand(self.repr_dim, self.repr_dim) * 0.5 - 0.3) for _ in range(self.k)]
        covs = [covs[k] @ covs[k].T for k in range(self.k)]
        self.weights = torch.div(torch.ones((self.k,)), self.k)
        self.components = [
            MultivariateNormal(loc=mus[i], covariance_matrix=covs[i])
            for i in range(self.k)
        ]

        self.samples = self._sample(n_samples)

    def _sample(self, n_samples):
        z = np.random.choice(self.k, n_samples, p=self.weights.numpy())
        samples = torch.stack([self.components[zi].rsample() for zi in z], dim=0)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: Union[List[int], Tensor]):
        return self.samples[idx]
