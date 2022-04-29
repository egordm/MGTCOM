from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any, Optional, Tuple

import igraph
import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from datasets.utils.temporal import TemporalNodeIndex
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecBatch
from ml.utils import HParams

try:
    import tch_geometric.tch_geometric as tch_native

    biased_tempo_random_walk = tch_native.biased_tempo_random_walk
except ImportError:
    biased_tempo_random_walk = None


class BiasType(Enum):
    Uniform = "uniform"
    Linear = "linear"
    Exponential = "exponential"


@dataclass
class CTDNESamplerParams(HParams):
    walk_length: int = 20
    """Length of the temporal random walk."""
    context_size: int = 10
    """The actual context size which is considered for positive samples. This parameter increases the effective 
    sampling rate by reusing samples across different source nodes.."""
    walks_per_node: int = 10
    """Number of random walks to start at each node."""
    num_neg_samples: int = 1
    """The number of negative samples to use for each positive sample."""
    walk_bias: BiasType = BiasType.Uniform
    """Distribution to use when choosing a random neighbour to walk through. 
    If set to 'Uniform', The initial edge is picked from a uniform distribution. 
    If set to 'Exponential' decaying probability, resulting in a bias towards shorter time gaps.
    """
    forward: bool = True
    """Whether to walk forward or backward in time."""
    retry_count: int = 10
    """Number of times to retry a random walk if it fails."""


class CTDNESampler(Sampler):
    def __init__(
            self,
            node_timestamps: Tensor, edge_index: Tensor, edge_timestamps: Tensor,
            hparams: CTDNESamplerParams = None,
            transform_meta: Callable[[Tensor], Any] = None,
    ) -> None:
        """
        The Node2Vec model from the
            `"node2vec: Scalable Feature Learning for Networks"
            <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
            length :obj:`walk_length` are sampled in a given graph, and node embeddings
            are learned via negative sampling optimization.

        Code from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html
        """
        super().__init__()

        if biased_tempo_random_walk is None:
            raise ImportError('`BallroomSampler` requires `tch_geometric`.')

        self.hparams = hparams or CTDNESamplerParams()
        assert self.hparams.walk_length >= self.hparams.context_size
        self.transform_meta = transform_meta

        self.walk_length = self.hparams.walk_length - 1

        self.temporal_index = TemporalNodeIndex().fit(node_timestamps, edge_index, edge_timestamps)
        self.num_nodes = len(node_timestamps)
        self.row_ptrs, self.col_indices, self.perm = tch_native.to_csr(edge_index, self.num_nodes)
        self.node_timestamps = node_timestamps
        self.edge_timestamps = edge_timestamps[self.perm]

    def sample(self, node_ids: Tensor) -> Node2VecBatch:
        pos_walks, neg_walks = self._pos_sample(node_ids), self._neg_sample(node_ids)
        walks = torch.cat([pos_walks, neg_walks], dim=0).view(-1)

        node_idx, perm = torch.unique(walks.view(-1), return_inverse=True)
        walks = perm.view(-1, self.hparams.context_size)
        pos_walks, neg_walks = walks[:pos_walks.shape[0]], walks[pos_walks.shape[0]:]

        node_meta = node_idx if self.transform_meta is None else self.transform_meta(node_idx)
        return Node2VecBatch(pos_walks, neg_walks, node_meta)

    def _pos_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.hparams.walks_per_node)

        node_timestamps = self.node_timestamps[batch]
        rw = biased_tempo_random_walk(
            self.row_ptrs, self.col_indices, self.node_timestamps, self.edge_timestamps,
            batch, node_timestamps, self.hparams.walk_length,
            self.hparams.walk_bias.value, self.hparams.forward, self.hparams.retry_count
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.hparams.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.hparams.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.hparams.walks_per_node * self.hparams.num_neg_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length - 1))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.hparams.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.hparams.context_size])
        return torch.cat(walks, dim=0)
