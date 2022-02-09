from typing import Any

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class LabelEntropy(Metric):
    def __init__(
            self,
            num_classes: int,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.add_state(
            name='label_counts',
            default=torch.zeros(num_classes, dtype=torch.int32),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, **__: Any) -> None:
        self.label_counts = torch.scatter_add(self.label_counts, 0, preds, torch.ones_like(preds, dtype=torch.int32))

    def compute(self) -> Any:
        counts = self.label_counts[self.label_counts > 0]
        probs = counts / counts.sum()
        return torch.sum(-probs * torch.log(probs))
