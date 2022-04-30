import torch
from torch import Tensor

from ml.utils import Metric, EPS


class IsometricLoss(torch.nn.Module):
    def __init__(self, metric: Metric = Metric.L2) -> None:
        super().__init__()
        self.sim_fn = metric.pairwise_sim_fn

    def forward(self, X: Tensor, r: Tensor, mus: Tensor):
        N = len(X)
        X_tag = X.repeat_interleave(len(mus), dim=0)
        mus_tag = mus.repeat(len(X), 1).view(-1, X.shape[1])
        r_tag = r.flatten()
        sim_tag = self.sim_fn(X_tag, mus_tag)
        # sim_tagz = -sim_tag
        sim_tagz = -torch.log(torch.sigmoid(sim_tag) + EPS)

        loss = (r_tag * sim_tagz).sum() / N

        return loss
