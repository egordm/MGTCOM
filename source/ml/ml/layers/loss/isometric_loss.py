import torch
from torch import Tensor

from ml.utils import Metric


class IsoGMMLoss(torch.nn.Module):
    def __init__(self, sim: Metric = Metric.L2) -> None:
        super().__init__()
        self.dist_fn = sim.pairwise_dist_fn

    def forward(self, X: Tensor, r: Tensor, mus: Tensor):


        X_tag = X.repeat_interleave(len(gmm), dim=0)
        mus_tag = gmm.mus.repeat(len(X), 1).view(-1, len(gmm))
        r_tag = r.flatten()
        dist_tag = self.dist_fn(X_tag, mus_tag).pow(2)
        loss = (r_tag * dist_tag).sum() / len(X)

        return loss, None  # , loss_cl
