import math
from typing import List, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter


class ClusterCohesionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def forward(self, q_l: torch.Tensor, q_r: torch.Tensor, label: torch.Tensor):
        sim = (self.cos_sim(q_l, q_r) + 1) / 2
        return torch.mean(torch.square(label - sim))


class NegativeEntropyRegularizer(torch.nn.Module):
    def forward(self, *args: List[torch.Tensor]):
        p = sum(map(lambda q: torch.sum(q, dim=0), args))
        p /= p.sum()
        ne = math.log(p.size(0)) + (p * torch.log(p)).sum()
        return ne


class HingeLoss(torch.nn.Module):
    def __init__(self, delta=1.0, agg_pos='mean', agg_neg='max') -> None:
        super().__init__()
        self.delta = delta
        self.agg_pos = agg_pos
        self.agg_neg = agg_neg

    def forward(
            self,
            emb: Tensor,
            inv_idx: Tensor,
            pos_edges: Tuple[Tensor, Tensor],
            neg_edges: Tuple[Tensor, Tensor],
    ):
        # Gather embeddings for edge nodes
        pc_emb = emb[inv_idx[pos_edges[0]], :]
        pp_emb = emb[inv_idx[pos_edges[1]], :]
        nc_emb = emb[inv_idx[neg_edges[0]], :]
        nn_emb = emb[inv_idx[neg_edges[1]], :]

        # Compute positive and negative distances
        p_d_full = torch.bmm(pc_emb.unsqueeze(1), pp_emb.unsqueeze(2)).view(-1)
        p_d = scatter(p_d_full, pos_edges[0], dim=0, reduce=self.agg_pos)

        n_d_full = torch.bmm(nc_emb.unsqueeze(1), nn_emb.unsqueeze(2)).view(-1)
        n_d = scatter(n_d_full, neg_edges[0], dim=0, reduce=self.agg_neg)

        # Compute loss
        loss = torch.clip(n_d - p_d + self.delta, min=0).mean()

        return loss
