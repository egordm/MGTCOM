import torch
from torch import Tensor

from ml.utils import Metric, EPS


class IsometricLoss(torch.nn.Module):
    def __init__(self, metric: Metric = Metric.L2) -> None:
        super().__init__()
        self.sim_fn = metric.pairwise_sim_fn
        self.dist_fn = metric.pairwise_dist_fn
        self.margin = 1.0

    def forward(self, X: Tensor, r: Tensor, mus: Tensor):
        z = r.argmax(dim=1)
        mu_i = mus[z]
        diff = self.dist_fn(X, mu_i).square()
        loss = diff.mean()
        return loss