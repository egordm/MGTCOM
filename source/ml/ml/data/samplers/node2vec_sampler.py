from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from ml.data.samplers.base import Sampler

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None


class Node2VecSampler(Sampler):
    def __init__(
            self,
            edge_index: Tensor,
            walk_length: int = 10, context_size: int = 4,
            walks_per_node: int = 5,
            num_neg_samples: int = 1,
            p=1, q=1, num_nodes=None
    ) -> None:
        """
        The Node2Vec model from the
            `"node2vec: Scalable Feature Learning for Networks"
            <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
            length :obj:`walk_length` are sampled in a given graph, and node embeddings
            are learned via negative sampling optimization.

        Code from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html


        :param num_nodes: The number of nodes in the graph
        :param edge_index: The edge indices of the graph
        :param walk_length: The walk length.
        :param context_size: The actual context size which is considered for positive samples. This parameter
            increases the effective sampling rate by reusing samples across different source nodes.
        :param walks_per_node: Number of random walks to start at each node, defaults to 5
        :param num_neg_samples: The number of negative samples to use for each positive sample, defaults to 1
        :param p: Likelihood of immediately revisiting a node in the walk. (default: :obj:`1`)
        :param q: Control parameter to interpolate between breadth-first strategy and depth-first strategy (default: :obj:`1`)
        """
        super().__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_neg_samples = num_neg_samples
        self.p = p
        self.q = q

        assert walk_length >= context_size

        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to('cpu')

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, node_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pos_walks, neg_walks = self._pos_sample(node_ids), self._neg_sample(node_ids)
        walks = torch.cat([pos_walks, neg_walks], dim=0).view(-1)

        node_idx, perm = torch.unique(walks.view(-1), return_inverse=True)
        walks = perm.view(-1, self.context_size)
        pos_walks, neg_walks = walks[:pos_walks.shape[0]], walks[pos_walks.shape[0]:]

        return pos_walks, neg_walks, node_idx

    def _pos_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.walks_per_node * self.num_neg_samples)

        rw = torch.randint(self.adj.sparse_size(0), (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
