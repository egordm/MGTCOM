import math
from typing import Union, List

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_scatter import scatter

from ml.layers.distance import pairwise_dotp


class HingeLoss(torch.nn.Module):
    def __init__(self, delta=1.0, agg_pos='mean', agg_neg='max') -> None:
        super().__init__()
        self.delta = delta
        self.agg_pos = agg_pos
        self.agg_neg = agg_neg

    def forward(
            self,
            emb: Tensor,
            data: Union[Data, HeteroData],
    ):
        if isinstance(data, HeteroData):
            nodes = data['n'].x
            pos_edges = data[('n', 'pos', 'n')].edge_index
            neg_edges = data[('n', 'neg', 'n')].edge_index
        else:
            nodes = data.x
            pos_edges = data.edge_index
            neg_edges = data.edge_index

        # Gather embeddings for edge nodes
        pc_emb = emb[nodes[pos_edges[0]], :]
        pp_emb = emb[nodes[pos_edges[1]], :]
        nc_emb = emb[nodes[neg_edges[0]], :]
        nn_emb = emb[nodes[neg_edges[1]], :]

        # Compute positive and negative similarity
        p_d_full = pairwise_dotp(pc_emb, pp_emb)
        p_d = scatter(p_d_full, pos_edges[0], dim=0, reduce=self.agg_pos)

        n_d_full = pairwise_dotp(nc_emb, nn_emb)
        n_d = scatter(n_d_full, neg_edges[0], dim=0, reduce=self.agg_neg)

        # Compute loss
        loss = torch.clip(n_d - p_d + self.delta, min=0).mean()

        return loss


class NegativeEntropyRegularizer(torch.nn.Module):
    def forward(self, *args: List[torch.Tensor]):
        """
        Averages all predictions into class based probability distribution and calculates negative entropy.
        See: https://en.wikipedia.org/wiki/Entropy#Information_theory
        The higher the `ne` value is the more unbalanced the class distribution is.
        :param args:
        :return:
        """
        p = sum(map(lambda q: torch.sum(q, dim=0), args))
        p /= p.sum()
        ne = math.log(p.size(0)) + (p * torch.log(p)).sum()
        return ne
