import torch
from torch import Tensor

from ml.utils import Metric


class IsometricLoss(torch.nn.Module):
    def __init__(self, metric: Metric = Metric.DOTP) -> None:
        super().__init__()
        metric = Metric.L2
        self.dist_fn = metric.pairwise_dist_fn

    def forward(self, X: Tensor, r: Tensor, mus: Tensor):
        N = len(X)
        X_tag = X.repeat_interleave(len(mus), dim=0)
        mus_tag = mus.repeat(len(X), 1).view(-1, X.shape[1])
        r_tag = r.flatten()
        dist_tag = self.dist_fn(X_tag, mus_tag).pow(2)
        loss = (r_tag * dist_tag).sum() / N

        return loss
