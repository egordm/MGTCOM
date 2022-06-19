from dataclasses import dataclass
from typing import Tuple, Callable, Any

import torch
from torch import Tensor
from typing_extensions import Self

from datasets.utils.temporal import TemporalNodeIndex
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams
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


class TemporalSampler(Sampler):
    def __init__(
            self,
            node_timestamps: Tensor, edge_index: Tensor, edge_timestamps: Tensor,
            window: Tuple[int, int],
            hparams: BallroomSamplerParams = None,
            transform_meta: Callable[[Tensor], Any] = None,
    ) -> None:
        super().__init__()

        if tempo_random_walk is None:
            raise ImportError('`BallroomSampler` requires `tch_geometric`.')

        self.hparams = hparams or BallroomSamplerParams()
        self.transform_meta = transform_meta
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

        node_meta = node_idx if self.transform_meta is None else self.transform_meta(node_idx)
        return Node2VecBatch(pos_walks, neg_walks, node_meta)

    def _pos_sample(self, node_ids: Tensor) -> Tensor:
        node_timestamps = self.temporal_index.node_to_timestamp(node_ids)

        # Infer timestamps for nodes that have no timestamp
        missing_timestamp_mask = (node_timestamps == NAN_TIMESTAMP).nonzero().flatten()
        if len(missing_timestamp_mask) > 0:
            node_timestamps[missing_timestamp_mask] = self._infer_missing_timestamps(node_ids[missing_timestamp_mask])

        # Consider multiple walks per node (for better coverage of its timewindow)
        batch = node_ids.repeat_interleave(self.hparams.walks_per_node * self.hparams.context_size, dim=0)
        batch_timestamps = node_timestamps.repeat_interleave(self.hparams.walks_per_node * self.hparams.context_size, dim=0)

        # Sample random neighborhoods given timestamp
        neighbor_idx = self.temporal_index.window_to_node(batch_timestamps, self.window)
        if (neighbor_idx == NAN_NODE_ID).any():
            neighbor_idx[neighbor_idx == NAN_NODE_ID] = batch[neighbor_idx == NAN_NODE_ID]

        walks = neighbor_idx.reshape(-1, self.hparams.context_size)
        return walks

    def _neg_sample(self, node_ids: Tensor) -> Tensor:
        batch = node_ids.repeat(self.hparams.walks_per_node * self.hparams.num_neg_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.hparams.context_size - 1))
        walks = torch.cat([batch.view(-1, 1), rw], dim=-1)

        return walks

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
