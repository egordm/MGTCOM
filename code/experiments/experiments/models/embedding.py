import torch

from torch_geometric.typing import Metadata
import torch_geometric.nn as tg_nn


def GraphSAGEEmbeddingModule(metadata: Metadata, repr_dim: int = 32, n_layers: int = 2) -> torch.nn.Module:
    components = []
    for i in range(n_layers - 1):
        components.extend([
            (tg_nn.SAGEConv((-1, -1), repr_dim), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True)
        ])

    embedding_module = tg_nn.Sequential('x, edge_index', [
        *components,
        (tg_nn.SAGEConv((-1, -1), repr_dim), 'x, edge_index -> x'),
    ])
    return tg_nn.to_hetero(embedding_module, metadata, aggr='mean')


class LinkPredictionModule(torch.nn.Module):
    def __init__(self, node_type: str, embedding_module: torch.nn.Module, dist='cosine'):
        super(LinkPredictionModule, self).__init__()
        self.embedding_module = embedding_module
        self.node_type = node_type

        if dist == 'cosine':
            self.dist = torch.nn.CosineSimilarity(dim=1)
        elif dist == 'euclidean':
            self.dist = torch.nn.PairwiseDistance(p=2)

        self.lin = torch.nn.Linear(1, 2)

    def forward(self, batch: torch.Tensor):
        batch_l, batch_r, label = batch
        batch_size = batch_l[self.node_type].batch_size

        emb_l = self.embedding_module(batch_l.x_dict, batch_l.edge_index_dict)[self.node_type][:batch_size]
        emb_r = self.embedding_module(batch_r.x_dict, batch_r.edge_index_dict)[self.node_type][:batch_size]
        dist = self.dist(emb_l, emb_r)
        logits = self.lin(torch.unsqueeze(dist, 1))

        return logits, dist, emb_l, emb_r
