import time
from dataclasses import dataclass, field
from typing import Callable, Any, List

import numpy as np
import scipy.sparse
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecBatch
from ml.utils import HParams


@dataclass
class CPGNNSamplerParams(HParams):
    k_hops: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_neg_samples: int = 1


class CPGNNSampler(Sampler):
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes=None,
        hparams: CPGNNSamplerParams = None,
        transform_meta: Callable[[Tensor], Any] = None,
    ) -> None:
        super().__init__()

        self.hparams = hparams or CPGNNSamplerParams()

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        self.transform_meta = transform_meta

        # Build adjacency matrix
        edge_index = edge_index.numpy()
        adj = scipy.sparse.csr_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(self.num_nodes, self.num_nodes),
        ).minimum(1.0)

        # Build k-hop adjacency matrix
        self.edge_index_ks = []
        for k in self.hparams.k_hops:
            adj_k = adj ** k
            edge_index_k = torch.from_numpy(np.stack(adj_k.nonzero(), axis=1)).long().t()
            self.edge_index_ks.append(edge_index_k)

    def __len__(self, k: int) -> int:
        return self.edge_index_ks[k].shape[1]

    def sampler(self, k: int, idx: Tensor) -> Node2VecBatch:
        edge_index = self.edge_index_ks[k]

        pos_edges = edge_index[:, idx]
        neg_edges = negative_sampling(
            edge_index, num_nodes=self.num_nodes,
            num_neg_samples=len(idx) * self.hparams.num_neg_samples,
        )
        edges = torch.cat([pos_edges, neg_edges], dim=0).view(-1)

        node_idx, perm = torch.unique(edges.view(-1), return_inverse=True)
        walks = perm.view(-1, 2)
        pos_edges, neg_edges = walks[:pos_edges.shape[0]], walks[neg_edges.shape[0]:]

        node_meta = node_idx if self.transform_meta is None else self.transform_meta(node_idx)
        return Node2VecBatch(pos_edges, neg_edges, node_meta)
