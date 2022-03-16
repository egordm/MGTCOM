from typing import Dict

import faiss
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from ml import igraph_from_hetero


class LouvainInitialization:
    def __init__(self, data: HeteroData):
        self.G, self.node_type_to_idx, self.edge_type_to_idx = igraph_from_hetero(data)
        _, self.type_counts = torch.tensor(self.G.vs['type'], dtype=torch.long).unique(return_counts=True)
        # self.segments =

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        comm = self.G.community_multilevel()
        k = len(comm)

        assignments = torch.tensor(comm.membership, dtype=torch.long)
        assignments_dict = assignments.split(self.type_counts.tolist())
        I = {
            node_type: assignments_dict[idx]
            for node_type, idx in self.node_type_to_idx.items()
        }
        _, counter = assignments.unique(return_counts=True)

        rep_dim = next(iter(emb_dict.values())).shape[1]
        centers = torch.zeros((k, rep_dim), dtype=torch.float)
        for node_type, assignment in I.items():
            centers = centers.index_add_(0, assignment, emb_dict[node_type])
        centers = centers / counter.view(-1, 1)

        return centers


class KMeansInitialization:
    def __init__(self, _data: HeteroData, k: int, verbose=False):
        self.k = k
        self.verbose = verbose

    def initialize(self, emb_dict: Dict[NodeType, Tensor]):
        rep_dim = next(emb_dict.values()).shape[1]
        kmeans = faiss.Kmeans(rep_dim, k=self.k, niter=20, verbose=self.verbose, nredo=10)

        emb = torch.cat(list(emb_dict.values()), dim=0)
        kmeans.train(emb.numpy())
        centers = torch.tensor(kmeans.centroids, dtype=torch.float)

        return centers
