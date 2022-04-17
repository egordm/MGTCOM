from typing import List, Union

import numpy as np
import torch
from tch_geometric.transforms.hgt_sampling import NAN_TIMESTAMP
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ml.utils.tensor import partition_values


class HeteroNodesDataset(Dataset):
    def __init__(self, data: HeteroData):
        self.data = data

        node_borders = np.cumsum([0] + [store.num_nodes for store in data.node_stores])
        self.node_ranges = list(zip(node_borders[:-1], node_borders[1:]))

    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx: Union[List[int], Tensor]):
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, dtype=torch.long)

        # Split idx into partitions
        partitions = partition_values(idx, self.node_ranges)
        idx_split = [idx - start for (idx, (start, _)) in zip(partitions, self.node_ranges)]

        # Extract nodes for each type
        node_index_dict = {
            node_type: idx
            for (node_type, idx) in zip(self.data.node_types, idx_split)
        }

        # Extract node timestamps
        node_timestamp_dict = None
        if self.temporal:
            node_timestamp_dict = {
                node_type: self.data[node_type].timestamp[idx] if hasattr(self.data[node_type], 'timestamp')
                else torch.full((len(idx),), NAN_TIMESTAMP)
                for (node_type, idx) in zip(self.data.node_types, idx_split)
            }

        data = HeteroData()
        for node_type in node_index_dict.keys():
            data[node_type].x = node_index_dict[node_type]

            if self.temporal:
                data[node_type].timestamp = node_timestamp_dict[node_type]

        return data
