from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeType

from datasets.transforms.random_node_split import RandomNodeSplit


class EvalNodeSplitTransform(BaseTransform):
    def __init__(
            self,
            split: str = "train_rest",
            num_splits: int = 1,
            num_val=0.1,
            num_test=0.1,
            force_resplit: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.num_splits = num_splits
        self.num_val = num_val
        self.num_test = num_test
        self.force_resplit = force_resplit

    def __call__(self, data: HeteroData):
        if 'train_mask' not in data or self.force_resplit:
            data = RandomNodeSplit(
                split=self.split,
                num_splits=self.num_splits,
                num_val=self.num_val,
                num_test=self.num_test,
                key=None,
            )(data)

        new_mask = {
            node_type: torch.zeros(num_nodes, dtype=torch.bool)
            for node_type, num_nodes in data.num_nodes_dict.items()
        }
        new_ids = {
            node_type: torch.full([num_nodes], -1, dtype=torch.long)
            for node_type, num_nodes in data.num_nodes_dict.items()
        }
        new_count = {
            node_type: 0
            for node_type in data.node_types
        }

        result = [
            self.split_masked(data, data.train_mask_dict, new_mask, new_ids, new_count),
            self.split_masked(data, data.val_mask_dict, new_mask, new_ids, new_count),
            self.split_masked(data, data.test_mask_dict, new_mask, new_ids, new_count)
        ]

        # for store in data.node_stores:
        #     store.eval_id = new_ids[store._key]

        return result

    def split_masked(
            self,
            data: HeteroData, mask_dict: Dict[NodeType, Tensor],
            new_mask, new_ids, new_count,
    ):
        for node_type in data.node_types:
            mask = mask_dict[node_type]
            num_nodes = mask.sum()
            new_mask[node_type] = torch.logical_or(mask, new_mask[node_type])
            new_ids[node_type][mask] = torch.arange(new_count[node_type], new_count[node_type] + num_nodes)
            new_count[node_type] += num_nodes

        result = data.clone()
        for node_type in data.node_types:
            ids = torch.arange(data[node_type].num_nodes)
            mask = new_mask[node_type]
            out_store = result[node_type]
            store = data[out_store._key]
            perm = torch.argsort(new_ids[node_type][mask])

            for key, value in store.items():
                if store.is_node_attr(key):
                    out_store[key] = value[mask][perm]
                elif isinstance(value, np.ndarray) and len(value) == store.num_nodes:
                    out_store[key] = value[mask][perm]

            out_store.id = ids[mask][perm]

        for out_store in result.edge_stores:
            store = data[out_store._key]
            src, _, dst = store._key
            mask_src = new_mask[src][store.edge_index[0, :]]
            mask_dst = new_mask[dst][store.edge_index[1, :]]
            mask = torch.logical_and(mask_src, mask_dst)

            for key, value in store.items():
                if key == 'edge_index':
                    out_store[key] = torch.stack([
                        new_ids[src][value[0, mask]],
                        new_ids[dst][value[1, mask]],
                    ])

                elif store.is_edge_attr(key):
                    out_store[key] = value[mask]
                elif isinstance(value, np.ndarray) and len(value) == store.num_edges:
                    out_store[key] = value[mask]

        return result
