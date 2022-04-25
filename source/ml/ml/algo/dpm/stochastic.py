from typing import Tuple

import torch
from torch import Tensor

from ml.algo.dpm.statistics import DPMMObs
from ml.utils import EPS


class DPMMObsMeanFilter:
    Ns: Tensor
    mus: Tensor
    covs: Tensor

    def __init__(self, k: int, repr_dim: int) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.Ns = torch.zeros(k, dtype=torch.float)
        self.mus = torch.zeros(k, repr_dim, dtype=torch.float)
        self.covs = torch.zeros(k, repr_dim, repr_dim, dtype=torch.float)

    def push(self, obs: DPMMObs) -> None:
        self.Ns += obs.Ns
        self.mus += obs.mus * obs.Ns.reshape(-1, 1)
        self.covs += obs.covs * obs.Ns.reshape(-1, 1, 1)

    def compute(self) -> DPMMObs:
        return DPMMObs(
            self.Ns,
            self.mus / (self.Ns.reshape(-1, 1) + EPS),
            self.covs / (self.Ns.reshape(-1, 1, 1) + EPS),
        )

    def reset(self, k: int) -> None:
        self.Ns = torch.zeros(k, dtype=torch.float)
        self.mus = torch.zeros(k, self.repr_dim, dtype=torch.float)
        self.covs = torch.zeros(k, self.repr_dim, self.repr_dim, dtype=torch.float)
