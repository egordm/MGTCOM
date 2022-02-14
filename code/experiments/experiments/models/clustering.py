from abc import abstractmethod
from typing import Optional, Union, Callable

import torch


def euclidean_cdist(x, y):
    return torch.cdist(x, y, p=2)


def cosine_cdist(x, y, dim=1):
    return torch.cosine_similarity(x.unsqueeze(dim), y.unsqueeze(dim-1), dim=dim+1)


class ClusteringModule(torch.nn.Module):
    def __init__(self, repr_dim: int, n_clusters: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.repr_dim = repr_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_assign(self, x: torch.Tensor) -> torch.Tensor:
        q = self(x)
        return q.argmax(dim=1)


class ExplicitClusteringModule(ClusteringModule):
    def __init__(
            self, repr_dim: int, n_clusters: int,
            centroids: Optional[torch.Tensor] = None,
            dist='cosine',
    ):
        super().__init__(repr_dim, n_clusters)
        self.dist = dist

        if centroids is None:
            initial_centroids = torch.zeros(self.n_clusters, self.repr_dim, dtype=torch.float)
            torch.nn.init.xavier_uniform_(initial_centroids)
        else:
            assert centroids.shape == (self.n_clusters, self.repr_dim)
            initial_centroids = centroids

        self.centroids = torch.nn.Parameter(initial_centroids)
        self.softmax = torch.nn.Softmax(dim=1)
        if dist == 'cosine':
            self.cdist_fn = cosine_cdist
        elif dist == 'euclidean':
            self.cdist_fn = euclidean_cdist
        else:
            raise ValueError(f'Unknown distance {dist}')

    def forward(self, batch: torch.Tensor):
        sim = self.cdist_fn(batch, self.centroids)
        if self.dist == 'euclidean':
            sim = -sim

        return self.softmax(sim)


class ImplicitClusteringModule(ClusteringModule):
    def __init__(
            self,
            repr_dim: int,
            n_clusters: int
    ):
        super().__init__(repr_dim, n_clusters)
        self.lin = torch.nn.Linear(repr_dim, n_clusters)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, batch: torch.Tensor):
        return self.activation(self.lin(batch))
