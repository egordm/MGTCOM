from typing import Tuple

import torch
from torch import Tensor


class MeanParams:
    Ns: Tensor
    mus: Tensor
    covs: Tensor

    def __init__(self, k: int, repr_dim: int) -> None:
        super().__init__()
        self.Ns = torch.zeros(k, dtype=torch.float)
        self.mus = torch.zeros(k, repr_dim, dtype=torch.float)
        self.covs = torch.zeros(k, repr_dim, repr_dim, dtype=torch.float)

    def push(self, N: Tensor, mus: Tensor, covs: Tensor) -> None:
        self.Ns += N
        self.mus += mus * N.reshape(-1, 1)
        self.covs += covs * N.reshape(-1, 1, 1)

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.Ns, self.mus / self.Ns.reshape(-1, 1), self.covs / self.Ns.reshape(-1, 1, 1)

    def reset(self):
        self.Ns = torch.zeros(self.Ns.shape, dtype=torch.float)
        self.mus = torch.zeros(self.mus.shape, dtype=torch.float)
        self.covs = torch.zeros(self.covs.shape, dtype=torch.float)
