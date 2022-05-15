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
        # return self.alt_loss(X, r, mus)
        #
        # z = r.argmax(dim=1)
        # mu_i = mus[z]
        # diff = self.dist_fn(X, mu_i)
        # loss = diff.mean()
        # return loss #TODO: which will we use?

        N = len(X)
        X_tag = X.repeat_interleave(len(mus), dim=0)
        mus_tag = mus.repeat(len(X), 1).view(-1, X.shape[1])
        r_tag = r.flatten()
        sim_tag = self.sim_fn(X_tag, mus_tag)
        # sim_tagz = -sim_tag
        sim_tagz = -torch.log(torch.sigmoid(sim_tag) + EPS)

        loss = (r_tag * sim_tagz).sum() / N
        # return self.alt_loss(X, r, mus)
        return loss

    def alt_loss(self, X: Tensor, r: Tensor, mus: Tensor):
        N = len(X)
        X_tag = X.unsqueeze(1)
        mus_tag = mus.unsqueeze(0)
        aff_tag = self.sim_fn(X_tag, mus_tag)
        z = r.argmax(dim=1)

        i_range = torch.arange(N)
        p_mask = torch.zeros_like(r, dtype=torch.bool, device=r.device)
        p_mask[i_range, z] = True

        p_aff = aff_tag[i_range, z].view(N, 1)
        n_aff = aff_tag[~p_mask].view(N, -1)

        diff = torch.relu(self.margin + n_aff - p_aff)# * r[~p_mask].view(N, -1).softmax(dim=1)
        # loss = diff.sum(dim=-1).mean()
        loss = diff.max(dim=1).values.mean()

        return loss

