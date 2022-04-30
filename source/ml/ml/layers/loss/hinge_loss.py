import torch
from torch import Tensor

from ml.layers.loss.skipgram_loss import SkipgramLoss, EPS
from ml.utils import Metric


class HingeLoss(SkipgramLoss):
    def __init__(
            self,
            metric: Metric = Metric.L2,
            margin: float = 1,
            adaptive: bool = False,
    ) -> None:
        super().__init__(metric)
        self.margin = margin
        self.adaptive = adaptive

    def check_compat(self):
        pass

    def forward(self, Z: Tensor, pos_walks: Tensor, neg_walks: Tensor) -> Tensor:
        pos_walks_Z = Z[pos_walks.view(-1)].view(*pos_walks.shape, Z.shape[-1])
        p_aff = self.affinity(pos_walks_Z).mean(dim=-1, keepdim=True)

        neg_walks_Z = Z[neg_walks.view(-1)].view(*neg_walks.shape, Z.shape[-1])
        n_aff = self.affinity(neg_walks_Z)
        if self.adaptive:
            n_aff = torch.max(n_aff, dim=-1).values

        diff = torch.relu(self.margin + n_aff - p_aff).mean(dim=-1)
        loss = diff.mean()
        return loss
