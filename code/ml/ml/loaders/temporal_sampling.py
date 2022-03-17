import random
import sys
from typing import Callable, Tuple, Dict

import torch
from tch_geometric.data import HeteroData
from tch_geometric.loader import CustomLoader
from torch_geometric.typing import NodeType
from torch import Tensor
from torch.utils.data import Dataset

from ml.utils import randint_range


class DummyDataset(Dataset):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


class TemporalSamplerLoader(CustomLoader):
    def __init__(
            self,
            data: HeteroData,
            window: Tuple[int, int],
            neighbor_sampler,
            num_neg: int = 3,
            repeat_count: int = 1,
            **kwargs
    ):
        self.data = data
        self.window = torch.tensor(window, dtype=torch.long)
        self.neighbor_sampler = neighbor_sampler
        self.num_neg = num_neg
        self.repeat_count = repeat_count

        self.node_type_to_idx = {
            node_type: idx
            for idx, node_type in enumerate(data.node_types)
        }

        attr_node_perm = []
        attr_node_type = []
        attr_node_timestamp = []

        for store in data.node_stores:
            if not hasattr(store, 'timestamp'):
                continue

            node_type = store._key
            perm = (store.timestamp != -1).nonzero().squeeze()
            attr_node_perm.append(perm)
            attr_node_type.append(torch.full_like(perm, self.node_type_to_idx[node_type]))
            attr_node_timestamp.append(store.timestamp[perm])

        for store in data.edge_stores:
            if not hasattr(store, 'timestamp'):
                continue

            edge_type = store._key
            (src, _, dst) = edge_type

            perm = (store.timestamp != -1).nonzero().squeeze()
            (row, col) = store.edge_index[:, perm]

            attr_node_perm.append(row)
            attr_node_type.append(torch.full_like(row, self.node_type_to_idx[src]))
            attr_node_timestamp.append(store.timestamp[perm])

            attr_node_perm.append(col)
            attr_node_type.append(torch.full_like(col, self.node_type_to_idx[dst]))
            attr_node_timestamp.append(store.timestamp[perm])

        attr_node_timestamp, node_perm = torch.cat(attr_node_timestamp, dim=0).sort(dim=0)
        attr_node_perm = torch.cat(attr_node_perm, dim=0)[node_perm]
        attr_node_type = torch.cat(attr_node_type, dim=0)[node_perm]

        self.time_index = torch.vstack([attr_node_timestamp, attr_node_type, attr_node_perm]) \
            .unique(dim=1, sorted=True)

        dataset_size = self.time_index.shape[1] * repeat_count
        super().__init__(DummyDataset(dataset_size), **kwargs)

    def sample(self, idx):
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, dtype=torch.long)

        # Construct query / center nodes
        idx = idx % self.time_index.shape[1]
        ctr_timestamps, ctr_node_types, ctr_nodes = self.time_index[:, idx]

        # Positive sample
        windows = (self.window.repeat((len(idx), 1)) + ctr_timestamps.unsqueeze(1)).t()
        window_from, window_to = torch.searchsorted(self.time_index[0, :], windows.reshape(-1)).view(2, -1)
        pos_ranges = window_to - window_from
        pos_idx = randint_range(pos_ranges, low=window_from)
        pos_timestamps, pos_node_types, pos_nodes = self.time_index[:, pos_idx]

        # Negative sample
        neg_ranges = self.time_index.shape[1] - pos_ranges

        neg_idx = randint_range(neg_ranges.repeat(self.num_neg))
        correction_idx = (neg_idx >= window_from.repeat(self.num_neg)).nonzero().squeeze()
        neg_idx[correction_idx] += pos_ranges[correction_idx % len(idx)]
        neg_timestamps, neg_node_types, neg_nodes = self.time_index[:, neg_idx]

        # Combine data
        nodes = torch.cat([ctr_nodes, pos_nodes, neg_nodes], dim=0)
        nodes_types = torch.cat([ctr_node_types, pos_node_types, neg_node_types], dim=0)
        nodes_timestamps = torch.cat([ctr_timestamps, pos_timestamps, neg_timestamps], dim=0)

        # Extract unique nodes for sampling and save their new indices
        nodes_dict: Dict[NodeType, Tensor] = {}
        nodes_timestamps_dict: Dict[NodeType, Tensor] = {}
        offset = 0
        for node_type, i in self.node_type_to_idx.items():
            type_idx = torch.where(nodes_types == i)[0]
            nodes_dict[node_type], perm = nodes[type_idx].unique(return_inverse=True)
            nodes_timestamps_dict[node_type] = nodes_timestamps[type_idx]
            nodes[type_idx] = perm + offset
            offset += len(type_idx)

        # Sample nodes
        samples = self.neighbor_sampler(nodes_dict, nodes_timestamps_dict)

        # Construct contrastive graph
        data = HeteroData()
        data['n'].x = nodes
        data['n'].timestamp = nodes_timestamps
        data['n', 'pos', 'n'].edge_index = torch.vstack([
            torch.arange(0, len(ctr_nodes), dtype=torch.long),
            torch.arange(len(ctr_nodes), len(ctr_nodes) + len(pos_nodes), dtype=torch.long),
        ])
        data['n', 'neg', 'n'].edge_index = torch.vstack([
            torch.arange(0, len(ctr_nodes), dtype=torch.long).repeat(self.num_neg),
            torch.arange(len(ctr_nodes) + len(pos_nodes), len(nodes), dtype=torch.long),
        ])

        return samples, data, list(self.node_type_to_idx.keys())
