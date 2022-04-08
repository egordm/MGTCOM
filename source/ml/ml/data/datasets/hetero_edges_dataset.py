from typing import List, Union

import numpy as np
import torch
from tch_geometric.data.subgraph import subgraph_from_edgelist
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ml.utils.tensor import partition_values


class HeteroEdgesDataset(Dataset):
    def __init__(self, data: HeteroData, temporal: bool = False):
        self.data = data
        self.temporal = temporal

        edge_borders = np.cumsum([0] + [store.num_edges for store in data.edge_stores])
        self.edge_ranges = list(zip(edge_borders[:-1], edge_borders[1:]))
        # TODO: add auxilary subindex to keep indices for different dataset partitions

    def __len__(self):
        return self.data.num_edges

    def __getitem__(self, idx: Union[List[int], Tensor]):
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, dtype=torch.long)

        # Split idx into partitions
        partitions = partition_values(idx, self.edge_ranges)
        idx_split = [idx - start for (idx, (start, _)) in zip(partitions, self.edge_ranges)]

        # Extract edges for each type
        edge_index_dict = {
            edge_type: self.data[edge_type].edge_index[:, idx]
            for (edge_type, idx) in zip(self.data.edge_types, idx_split)
        }

        # Extract edge timestamps
        edge_timestamp_dict = None
        if self.temporal:
            edge_timestamp_dict = {
                edge_type: self.data[edge_type].timestamp[idx]
                for (edge_type, idx) in zip(self.data.edge_types, idx_split)
            }

        # Transform edges to pos subgraph
        pos_data = subgraph_from_edgelist(
            edge_index_dict,
            edge_attrs=dict(edge_timestamp_dict=edge_timestamp_dict) if self.temporal else None
        )

        return pos_data
