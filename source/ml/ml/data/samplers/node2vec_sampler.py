from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from ml.data.samplers.base import Sampler
from ml.utils import HParams

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

Node2VecBatch = NamedTuple('Node2VecBatch', [("pos_walks", Tensor), ("neg_walks", Tensor), ("node_meta", Any)])


@dataclass
class Node2VecSamplerParams(HParams):
    walk_length: int = 10
    """Length of the random walk."""
    context_size: int = 4
    """The actual context size which is considered for positive samples. This parameter increases the effective 
    sampling rate by reusing samples across different source nodes.."""
    walks_per_node: int = 5
    """Number of random walks to start at each node."""
    num_neg_samples: int = 1
    """The number of negative samples to use for each positive sample."""
    p: float = 1
    """Likelihood of immediately revisiting a node in the walk."""
    q: float = 1
    """Control parameter to interpolate between breadth-first strategy and depth-first strategy."""


class Node2VecSampler(Sampler):
    def __init__(self, edge_index: Tensor, num_nodes=None, hparams: Node2VecSamplerParams = None) -> None:
        """
        The Node2Vec model from the
            `"node2vec: Scalable Feature Learning for Networks"
            <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
            length :obj:`walk_length` are sampled in a given graph, and node embeddings
            are learned via negative sampling optimization.

        Code from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html
        """
        super().__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        self.hparams = hparams or Node2VecSamplerParams()
        assert self.hparams.walk_length >= self.hparams.context_size

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        self.walk_length = self.hparams.walk_length - 1

        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to('cpu')

    def sample(self, node_ids: Tensor) -> Node2VecBatch:
        pos_walks, neg_walks = self._pos_sample(node_ids), self._neg_sample(node_ids)
        walks = torch.cat([pos_walks, neg_walks], dim=0).view(-1)

        node_idx, perm = torch.unique(walks.view(-1), return_inverse=True)
        walks = perm.view(-1, self.hparams.context_size)
        pos_walks, neg_walks = walks[:pos_walks.shape[0]], walks[pos_walks.shape[0]:]

        return Node2VecBatch(pos_walks, neg_walks, node_idx)

    def _pos_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.hparams.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.hparams.p, self.hparams.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.hparams.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.hparams.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.hparams.walks_per_node * self.hparams.num_neg_samples)

        rw = torch.randint(self.adj.sparse_size(0), (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.hparams.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.hparams.context_size])
        return torch.cat(walks, dim=0)
