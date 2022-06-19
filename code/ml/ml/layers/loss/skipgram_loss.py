from pathlib import Path

import torch
from torch import Tensor

from ml.utils import Metric
from shared import get_logger

EPS = 1e-15

logger = get_logger(Path(__file__).stem)


class SkipgramLoss(torch.nn.Module):
    def __init__(self, metric: Metric = Metric.DOTP) -> None:
        super().__init__()
        self.metric = metric
        self.sim_fn = metric.pairwise_sim_fn
        self.check_compat()

    def check_compat(self):
        if self.metric != Metric.DOTP:
            logger.warning('Skipgram loss is only compatible with dot product similarity. Otherwise, results may vary.')

    def affinity(self, walks_Z: Tensor):
        head, rest = walks_Z[:, 0].unsqueeze(dim=1), walks_Z[:, 1:]
        sim = self.sim_fn(head, rest)
        return sim

    def forward(self, Z: Tensor, pos_walks: Tensor, neg_walks: Tensor) -> Tensor:
        pos_walks_Z = Z[pos_walks.view(-1)].view(*pos_walks.shape, Z.shape[-1])
        p_aff = self.affinity(pos_walks_Z).view(-1)
        p_loss = -torch.log(torch.sigmoid(p_aff) + EPS).mean()

        neg_walks_Z = Z[neg_walks.view(-1)].view(*neg_walks.shape, Z.shape[-1])
        n_aff = self.affinity(neg_walks_Z).view(-1)
        # n_loss = -torch.log(1 - torch.sigmoid(n_aff) + EPS).mean()
        n_loss = -torch.log(torch.sigmoid(-n_aff) + EPS).mean()

        return p_loss + n_loss
