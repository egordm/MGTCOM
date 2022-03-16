from typing import Optional

import torch.nn

from ml.layers import pairwise_dotp
from ml.layers.distance import pairwise_cosine, pairwise_euclidean_sim


class ExplicitClusteringModule(torch.nn.Module):
    def __init__(
            self, repr_dim: int, n_clusters: int,
            initial_centroids: Optional[torch.Tensor] = None,
            sim='dotp',
    ):
        super().__init__()
        self.sim = sim

        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)
        self.softmax = torch.nn.Softmax(dim=1)

        if initial_centroids:
            self.centroids.weight.data = initial_centroids

        if sim == 'dotp':
            self.sim_fn = pairwise_dotp
        elif sim == 'cosine':
            self.sim_fn = pairwise_cosine
        elif sim == 'euclidean':
            self.sim_fn = pairwise_euclidean_sim
        else:
            raise ValueError(f'Unknown similarity function {sim}')

    def forward(self, batch: torch.Tensor):
        sim = self.sim_fn(batch.unsqueeze(1), self.centroids.weight.unsqueeze(0))
        return self.softmax(sim)

    def assign(self, batch: torch.Tensor):
        q = self.forward(batch)
        return q.argmax(dim=-1)

    def reinit(self, centers: torch.Tensor):
        device = self.centroids.weight.data.device
        self.centroids = torch.nn.Embedding.from_pretrained(centers, freeze=False).to(device)

    @property
    def n_clusters(self):
        return self.centroids.weight.shape[0]
