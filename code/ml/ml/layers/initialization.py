from typing import Dict

import faiss
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_scatter import scatter_mean


from ml import igraph_from_hetero


class LouvainInitialization:
    def __init__(self, data: HeteroData):
        self.node_types = data.node_types
        self.G, self.node_type_to_idx, self.edge_type_to_idx = igraph_from_hetero(data)
        _, self.type_counts = torch.tensor(self.G.vs['type'], dtype=torch.long).unique(return_counts=True)
        # self.segments =

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        comm = self.G.community_multilevel()
        k = len(comm)

        embs = torch.cat([
            emb_dict[node_type]
            for node_type in self.node_types
        ], dim=0)
        assignments = torch.tensor(comm.membership, dtype=torch.long)
        centers = scatter_mean(embs, assignments, dim=0, dim_size=k)

        return centers


class KMeansInitialization:
    def __init__(self, _data: HeteroData, k: int, verbose=False):
        self.k = k
        self.verbose = verbose

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        rep_dim = next(iter(emb_dict.values())).shape[1]
        kmeans = faiss.Kmeans(rep_dim, k=self.k, niter=20, verbose=self.verbose, nredo=10)

        emb = torch.cat(list(emb_dict.values()), dim=0)
        kmeans.train(emb.numpy())
        centers = torch.tensor(kmeans.centroids, dtype=torch.float)

        return centers
