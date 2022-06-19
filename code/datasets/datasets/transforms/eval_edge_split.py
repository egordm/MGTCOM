from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomNodeSplit, BaseTransform
from torch_geometric.typing import NodeType

from datasets.transforms.random_edge_split import RandomEdgeSplit


class EvalEdgeSplitTransform(BaseTransform):
    def __init__(
        self,
        num_val=0.1,
        num_test=0.1,
        key_prefix: str = "",
        force_resplit: bool = False,
    ) -> None:
        super().__init__()
        self.num_val = num_val
        self.num_test = num_test
        self.key_prefix = key_prefix
        self.force_resplit = force_resplit

    def __call__(self, data: HeteroData):
        train_key = self.key_prefix + 'train_mask'
        val_key = self.key_prefix + 'val_mask'
        test_key = self.key_prefix + 'test_mask'

        if train_key not in data or self.force_resplit:
            data = RandomEdgeSplit(
                num_val=self.num_val,
                num_test=self.num_test,
                key_prefix=self.key_prefix,
                inplace=True,
            )(data)

        train_mask = getattr(data, f'{train_key}_dict')
        val_mask = getattr(data, f'{val_key}_dict')
        test_mask = getattr(data, f'{test_key}_dict')

        val_edges_mask = {
            edge_type: train_mask[edge_type]
            for edge_type, mask in val_mask.items()
        }
        test_edges_mask = {
            edge_type: torch.logical_or(train_mask[edge_type], val_mask[edge_type])
            for edge_type, mask in test_mask.items()
        }

        result = [
            self.split_masked(data, train_mask),
            self.split_masked(data, val_edges_mask),
            self.split_masked(data, test_edges_mask)
        ]

        return result

    def split_masked(self, data: HeteroData, mask_dict: Dict[NodeType, Tensor]):
        out = data.clone()

        for out_store in out.edge_stores:
            edge_type = out_store._key
            store = data[edge_type]
            mask = mask_dict[edge_type]

            for key, value in store.items():
                if key == 'edge_index':
                    out_store[key] = value[:, mask]

                elif store.is_edge_attr(key):
                    out_store[key] = value[mask]
                elif isinstance(value, np.ndarray) and len(value) == store.num_edges:
                    out_store[key] = value[mask]

        return out
