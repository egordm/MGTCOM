from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Self

from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecBatch
from ml.utils import HParams, randint_range

try:
    import tch_geometric.tch_geometric as tch_native

    tempo_random_walk = tch_native.tempo_random_walk
except ImportError:
    tempo_random_walk = None

NAN_TIMESTAMP = -1
NAN_NODE_ID = -1


@dataclass
class BallroomSamplerParams(HParams):
    walk_length: int = 5
    """Length of the temporal random walk."""
    context_size: int = 4
    """The actual context size which is considered for positive samples. This parameter increases the effective 
    sampling rate by reusing samples across different source nodes.."""
    walks_per_node: int = 5
    """Number of random walks to start at each node. (i.e. number of partners per node)"""
    num_neg_samples: int = 1


class BallroomSampler(Sampler):
    def __init__(
            self,
            node_timestamps: Tensor, edge_index: Tensor, edge_timestamps: Tensor,
            window: Tuple[int, int],
            hparams: BallroomSamplerParams = None
    ) -> None:
        super().__init__()

        if tempo_random_walk is None:
            raise ImportError('`BallroomSampler` requires `tch_geometric`.')

        self.hparams = hparams or BallroomSamplerParams()
        self.window = window
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

        return Node2VecBatch(pos_walks, neg_walks, node_idx)

    def _pos_sample(self, node_ids: Tensor) -> Tensor:
        node_timestamps = self.temporal_index.node_to_timestamp(node_ids)

        # Infer timestamps for nodes that have no timestamp
        missing_timestamp_mask = (node_timestamps == NAN_TIMESTAMP).nonzero().flatten()
        if len(missing_timestamp_mask) > 0:
            node_timestamps[missing_timestamp_mask] = self._infer_missing_timestamps(node_ids[missing_timestamp_mask])

        # Consider multiple walks per node (for better coverage of its timewindow)
        batch = node_ids.repeat_interleave(self.hparams.walks_per_node, dim=0)
        batch_timestamps = node_timestamps.repeat_interleave(self.hparams.walks_per_node, dim=0)

        # Sample random neighborhoods given timestamp
        neighbor_idx = self.temporal_index.window_to_node(batch_timestamps, self.window)

        # Do temporal random walk for all the neighborhoods
        neigh_walks, _ = self._temporal_random_walk(neighbor_idx, batch_timestamps)
        contexts = neigh_walks.reshape(-1, self.hparams.walks_per_node * self.walk_length)
        contexts = contexts[:, torch.randperm(self.hparams.walks_per_node * self.walk_length)]\
            .view(-1, self.walk_length)
        contexts = torch.cat([batch.reshape(-1, 1), contexts], dim=1)

        # Construct context from the walks
        walks = [] # In total there should be (walks_per_node * (walk_length - context_size)) walks
        num_walks_per_rw = 1 + self.walk_length - self.hparams.context_size
        for j in range(num_walks_per_rw):
            walks.append(contexts[:, j:j + self.hparams.context_size])

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

    def _infer_missing_timestamps(self, node_ids: Tensor) -> Tensor:
        _, walks_timestamps = self._temporal_random_walk(node_ids, torch.full(node_ids.shape, NAN_TIMESTAMP))
        n_timestamps = (walks_timestamps != NAN_TIMESTAMP).sum(dim=1)
        timestamp_idx = randint_range(n_timestamps.clamp(min=1))

        node_timestamps = torch.full(node_ids.shape, NAN_TIMESTAMP)
        for i, (n, ti) in enumerate(zip(n_timestamps, timestamp_idx)):
            if n > 0:
                node_timestamps[i] = walks_timestamps[i, walks_timestamps[i] != NAN_TIMESTAMP][ti]

        return node_timestamps

    def _temporal_random_walk(self, node_ids: Tensor, node_timestamps: Tensor) -> Tuple[Tensor, Tensor]:
        return tempo_random_walk(
            self.row_ptrs, self.col_indices, self.node_timestamps, self.edge_timestamps,
            node_ids, node_timestamps, self.walk_length, self.window
        )


class TemporalNodeIndex:
    node_ids: Tensor
    node_timestamps: Tensor
    node_ids_rev: Tensor
    node_timestamps_rev: Tensor

    def __init__(self) -> None:
        super().__init__()

    def fit(self, node_timestamps: Tensor, edge_index: Tensor, edge_timestamps: Tensor) -> Self:
        # Combine node timestamps from node and edge data
        node_perm_index = torch.cat([
            torch.arange(len(node_timestamps)),
            edge_index[0, :],
            edge_index[1, :]
        ], dim=0)
        node_timestamp_index = torch.cat([
            node_timestamps,
            edge_timestamps,
            edge_timestamps,
        ])

        # Filter away all the nodes with no timestamp
        node_perm_index = node_perm_index[node_timestamp_index != NAN_TIMESTAMP]
        node_timestamp_index = node_timestamp_index[node_timestamp_index != NAN_TIMESTAMP]

        # Select for unique timestamps per node and sort
        node_index = torch.stack([node_perm_index, node_timestamp_index], dim=1)
        node_index = torch.unique(node_index, dim=0, sorted=True)

        self.node_ids = node_index[:, 0].contiguous()
        self.node_timestamps = node_index[:, 1].contiguous()

        # Build reverse index (timestamp -> node)
        self.node_timestamps_rev, perm = torch.sort(self.node_timestamps, descending=False)
        self.node_ids_rev = self.node_ids[perm]

        # TODO: remove after debugging
        self.node_ids = self.node_ids[13:]
        self.node_timestamps = self.node_timestamps[13:]

        return self

    def node_to_timestamp(self, node_ids: Tensor) -> Tensor:
        result_timestamps = torch.full_like(node_ids, NAN_TIMESTAMP, dtype=torch.long)

        # Find the range for each node in the index
        ranges = torch.searchsorted(
            self.node_ids, torch.cat([node_ids, node_ids + 1], dim=0), side='left'
        )
        range_from, range_to = ranges[:len(node_ids)], ranges[len(node_ids):]

        # Filter for nodes with a timestamp
        has_timestamp_mask = self.node_ids[range_from] == node_ids
        range_span = range_to[has_timestamp_mask] - range_from[has_timestamp_mask]

        # Pick a random timestamp for each node within its range
        timestamp_idx_picks = randint_range(range_span) + range_from[has_timestamp_mask]
        result_timestamps[has_timestamp_mask] = self.node_timestamps[timestamp_idx_picks]

        return result_timestamps

    def window_to_node(self, node_timestamps: Tensor, window: Tuple[int, int]) -> Tensor:
        result_ids = torch.full_like(node_timestamps, NAN_NODE_ID, dtype=torch.long)

        # Find the range for each timestamp in the index
        ranges = torch.searchsorted(
            self.node_timestamps_rev, torch.cat([node_timestamps + window[0], node_timestamps + window[1]], dim=0), side='left'
        )
        range_from, range_to = ranges[:len(node_timestamps)], ranges[len(node_timestamps):]

        # Filter for nodes with a timestamp
        has_node_mask = range_to - range_from > 0
        range_span = range_to[has_node_mask] - range_from[has_node_mask]

        # Pick a random timestamp for each node within its range
        node_idx_picks = randint_range(range_span) + range_from[has_node_mask]
        result_ids[has_node_mask] = self.node_ids_rev[node_idx_picks]

        return result_ids




