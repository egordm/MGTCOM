from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ml.utils import pairwise_sim_fn, pairwise_dist_fn

if TYPE_CHECKING:
    from ml.layers.dpm import GaussianMixtureModel


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class KLGMMLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, gmm: "GaussianMixtureModel", X: Tensor, r: Tensor):
        r = eps_norm(r)
        r_E = eps_norm(gmm.estimate_log_prob(X))
        loss = self.kl_div(torch.log(r), r_E)

        # loss_cl = (r.argmax(dim=-1) != r_E.argmax(dim=-1)).sum() / r.shape[1]

        return loss, None #, loss_cl


class IsoGMMLoss(torch.nn.Module):
    def __init__(self, sim='euclidean') -> None:
        super().__init__()
        self.dist_fn = pairwise_dist_fn(sim)

    def forward(self, gmm: "GaussianMixtureModel", X: Tensor, r: Tensor):
        X_tag = X.repeat_interleave(len(gmm), dim=0)
        mus_tag = gmm.mus.repeat(len(X), 1).view(-1, len(gmm))
        r_tag = r.flatten()
        dist_tag = self.dist_fn(X_tag, mus_tag).pow(2)
        loss = (r_tag * dist_tag).sum() / len(X)

        return loss, None #, loss_cl
