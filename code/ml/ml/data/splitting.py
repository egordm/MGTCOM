from typing import Union, Optional, List

import torch
from torch import Tensor
from copy import copy

from torch_geometric.typing import EdgeType
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage

from torch_geometric.transforms import BaseTransform


class LinkSplitter(BaseTransform):
    def __init__(
            self,
            num_val: Union[int, float] = 0.1,
            num_test: Union[int, float] = 0.2,
            edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.edge_types = edge_types

    def __call__(self, data):
        edge_types = self.edge_types

        result_train, result_val, result_test = copy(data), copy(data), copy(data)
        if isinstance(data, HeteroData):
            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to"
                    "be specified when operating on 'HeteroData' objects")

            stores = [data[edge_type] for edge_type in edge_types]
            stores_train = [result_train[edge_type] for edge_type in edge_types]
            stores_val = [result_val[edge_type] for edge_type in edge_types]
            stores_test = [result_test[edge_type] for edge_type in edge_types]
        else:
            stores = [data]
            stores_train = [result_train]
            stores_val = [result_val]
            stores_test = [result_test]

        for item in zip(edge_types, stores, stores_train, stores_val, stores_test):
            edge_type, store, train_store, val_store, test_store = item

            edge_index = store.edge_index
            perm = torch.randperm(edge_index.size(1), device=edge_index.device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())

            num_train = perm.numel() - num_val - num_test
            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")

            train_idx = perm[:num_train]
            val_idx = perm[:num_train + num_val]
            test_idx = perm[:]

            self._split(store, train_idx, train_store)
            self._split(store, val_idx, val_store)
            self._split(store, test_idx, test_store)

            edge_partitions = torch.zeros(edge_index.size(1), dtype=torch.uint8, device=edge_index.device)
            edge_partitions[train_idx] = 0
            edge_partitions[perm[num_train:num_train + num_val]] = 1
            edge_partitions[perm[num_train + num_val:]] = 2
            train_store.edge_partitions = edge_partitions[train_idx]
            val_store.edge_partitions = edge_partitions[val_idx]
            test_store.edge_partitions = edge_partitions[test_idx]

        return result_train, result_val, result_test

    def _split(
            self,
            in_store: EdgeStorage,
            index: Tensor,
            out_store: EdgeStorage,
    ):
        for key, value in in_store.items():
            if key == 'edge_index':
                continue

            if in_store.is_edge_attr(key):
                out_store[key] = value[index]

        out_store.edge_index = in_store.edge_index[:, index]
