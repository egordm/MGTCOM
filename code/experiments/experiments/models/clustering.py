from abc import abstractmethod
from typing import Optional, Union, Callable

import torch


def euclidean_cdist(x, y):
    return torch.cdist(x, y, p=2)


def cosine_cdist(x, y):
    return torch.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)


class ClusteringModule(torch.nn.Module):
    def __init__(self, rep_dim: int, n_clusters: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.rep_dim = rep_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_assign(self, x: torch.Tensor) -> torch.Tensor:
        q = self(x)
        return q.argmax(dim=1)


class ExplicitClusteringModule(ClusteringModule):
    def __init__(
            self, rep_dim: int, n_clusters: int,
            centroids: Optional[torch.Tensor] = None,
            cdist_fn=euclidean_cdist,
    ):
        super().__init__(rep_dim, n_clusters)

        if centroids is None:
            initial_centroids = torch.zeros(self.n_clusters, self.rep_dim, dtype=torch.float)
            torch.nn.init.xavier_uniform_(initial_centroids)
        else:
            assert centroids.shape == (self.n_clusters, self.rep_dim)
            initial_centroids = centroids

        self.centroids = torch.nn.Parameter(initial_centroids)
        self.softmax = torch.nn.Softmax(dim=1)
        self.cdist_fn = cdist_fn

    def forward(self, batch: torch.Tensor):
        sim = self.cdist_fn(batch, self.cluster_centers)
        return self.softmax(sim)


class ImplicitClusteringModule(ClusteringModule):
    def __init__(
            self,
            rep_dim: int,
            n_clusters: int
    ):
        super().__init__(rep_dim, n_clusters)
        self.lin = torch.nn.Linear(rep_dim, n_clusters)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, batch: torch.Tensor):
        return self.activation(self.lin(batch))
