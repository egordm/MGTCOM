from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ml.layers.dpm import GaussianMixtureModel


def eps_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x + eps) / (x + eps).sum(dim=-1, keepdim=True)


class KLGMMLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, gmm: "GaussianMixtureModel", x: Tensor, r: Tensor):
        r = eps_norm(r)
        r_E = eps_norm(gmm.estimate_log_prob(x))
        loss = self.kl_div(torch.log(r), r_E)

        return loss


class IsoGMMLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gmm: "GaussianMixtureModel", X: Tensor, r: Tensor):
        x_tag = X.repeat(1, gmm.tot_components()).view(-1, gmm.repr_dim)
        mus_tag = gmm.mus.repeat(X.shape[0], 1)
        r_tag = r.flatten()
        loss = torch.mean(r_tag * (x_tag - mus_tag).norm(dim=1).pow(2))

        return loss
