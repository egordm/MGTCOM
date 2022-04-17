from dataclasses import dataclass, field
from typing import Optional, List

import torch

from ml.utils import HParams


@dataclass
class FCNetParams(HParams):
    repr_dim: int = 32
    hidden_dim: Optional[List[int]] = field(default_factory=list)


class FCNet(torch.nn.Module):
    def __init__(self, in_dim: int, hparams: FCNetParams = None) -> None:
        super().__init__()
        self.in_dim = in_dim

        layers = []
        for hidden_dim in hparams.hidden_dim:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hparams.repr_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)
