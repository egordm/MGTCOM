from abc import abstractmethod
from typing import Dict

import faiss
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_scatter import scatter_mean
from pytorch_lightning.utilities.cli import _Registry

from ml import igraph_from_hetero

INITIALIZER_REGISTRY = _Registry()


class BaseInitializer:
    def __init__(self, data: HeteroData) -> None:
        super().__init__()
        self.data = data

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        pass


@INITIALIZER_REGISTRY
class RandomInitializer(BaseInitializer):
    def __init__(self, data: HeteroData, k: int) -> None:
        super().__init__()
        self.data = data
        self.k = k

    @property
    def n_clusters(self) -> int:
        return self.k


@INITIALIZER_REGISTRY
class LouvainInitializer(BaseInitializer):
    def __init__(self, data: HeteroData):
        super().__init__(data)
        self.G, self.node_type_to_idx, self.edge_type_to_idx = igraph_from_hetero(data)
        _, self.type_counts = torch.tensor(self.G.vs['type'], dtype=torch.long).unique(return_counts=True)
        self.comm = self.G.community_multilevel()

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        k = len(self.comm)

        embs = torch.cat([
            emb_dict[node_type]
            for node_type in self.data.node_types
        ], dim=0)
        assignments = torch.tensor(self.comm.membership, dtype=torch.long)
        centers = scatter_mean(embs, assignments, dim=0, dim_size=k)

        return centers

    @property
    def n_clusters(self) -> int:
        return len(self.comm)


class KMeansInitializer(BaseInitializer):
    def __init__(self, data: HeteroData, k: int, verbose=False):
        super().__init__(data)
        self.k = k
        self.verbose = verbose

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        rep_dim = next(iter(emb_dict.values())).shape[1]
        kmeans = faiss.Kmeans(rep_dim, k=self.k, niter=20, verbose=self.verbose, nredo=10)

        emb = torch.cat(list(emb_dict.values()), dim=0)
        kmeans.train(emb.numpy())
        centers = torch.tensor(kmeans.centroids, dtype=torch.float)

        return centers

    @property
    def n_clusters(self) -> int:
        return self.k
