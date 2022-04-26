from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Self

from datasets.utils.tensor import randint_range

NAN_TIMESTAMP = -1
NAN_NODE_ID = -1


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
            self.node_timestamps_rev, torch.cat([node_timestamps + window[0], node_timestamps + window[1]], dim=0),
            side='left'
        )
        range_from, range_to = ranges[:len(node_timestamps)], ranges[len(node_timestamps):]

        # Filter for nodes with a timestamp
        has_node_mask = range_to - range_from > 0
        range_span = range_to[has_node_mask] - range_from[has_node_mask]

        # Pick a random timestamp for each node within its range
        node_idx_picks = randint_range(range_span) + range_from[has_node_mask]
        result_ids[has_node_mask] = self.node_ids_rev[node_idx_picks]

        return result_ids
