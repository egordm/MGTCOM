from typing import Union, Optional, Dict

import torch
from torch import Tensor


class SubsampleTransform:
    perm: Optional[Tensor]

    def __init__(self, max_points: int = 1000) -> None:
        super().__init__()
        self.max_points = max_points
        self.perm = None

    def fit(self, X: Union[Tensor, int]):
        N = X if isinstance(X, int) else len(X)
        if N > self.max_points and self.max_points > 0:
            self.perm = torch.randperm(N)[:self.max_points]
        else:
            self.perm = torch.arange(N)
        return self

    def transform(self, X: Tensor) -> Tensor:
        if self.perm is None:
            self.fit(X)

        return X[self.perm]


class SubsampleDictTransform:
    perm: Optional[Dict[str, Tensor]] = None

    def __init__(self, max_points: int = 1000) -> None:
        super().__init__()
        self.max_points = max_points

    def fit(self, X: Union[Dict[str, Tensor], Dict[str, int]]):
        N = 0
        N_dict = {}
        for key, value in X.items():
            if isinstance(value, int):
                N += value
                N_dict[key] = value
            else:
                N += len(value)
                N_dict[key] = len(value)

        self.perm = {}
        if self.max_points > 0 and N > self.max_points:
            for key, value in X.items():
                self.perm[key] = torch.randperm(N_dict[key])[:int(N_dict[key] / float(self.max_points))]
        else:
            for key, value in X.items():
                self.perm[key] = torch.arange(N_dict[key])

        return self

    def transform(self, X: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.perm is None:
            self.fit(X)

        return {
            key: value[self.perm[key]]
            for key, value in X.items()
        }
