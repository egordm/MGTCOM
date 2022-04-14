import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter

from ml.utils.distance import Metric


class HingeLoss(torch.nn.Module):
    def __init__(self, delta=1.0, agg_pos='mean', agg_neg='max', sim: Metric = Metric.L2) -> None:
        super().__init__()
        self.delta = delta
        self.agg_pos = agg_pos
        self.agg_neg = agg_neg

        self.sim_fn = sim.pairwise_sim_fn

    def forward(
            self,
            emb: Tensor,
            samples: HeteroData,
    ):
        nodes = samples['n'].x
        pos_edges = samples[('n', 'pos', 'n')].edge_index
        neg_edges = samples[('n', 'neg', 'n')].edge_index

        # Gather embeddings for edge nodes
        pc_emb = emb[nodes[pos_edges[0]], :]
        pp_emb = emb[nodes[pos_edges[1]], :]
        nc_emb = emb[nodes[neg_edges[0]], :]
        nn_emb = emb[nodes[neg_edges[1]], :]

        # Compute positive and negative similarity
        p_d_full = self.sim_fn(pc_emb, pp_emb)
        p_d = scatter(p_d_full, pos_edges[0], dim=0, reduce=self.agg_pos)

        n_d_full = self.sim_fn(nc_emb, nn_emb)
        n_d = scatter(n_d_full, neg_edges[0], dim=0, reduce=self.agg_neg)

        # Compute loss
        loss = torch.clip(n_d - p_d + self.delta, min=0).mean()

        return loss
