from typing import Union, Any, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType


class RandomEdgeSplit(BaseTransform):
    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.1,
        key_prefix: str = "",
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.num_val = num_val
        self.num_test = num_test
        self.key_prefix = key_prefix
        self.inplace = inplace

    def __call__(self, data: HeteroData) -> HeteroData:
        data = data.clone() if not self.inplace else data

        train_key = self.key_prefix + 'train_mask'
        val_key = self.key_prefix + 'val_mask'
        test_key = self.key_prefix + 'test_mask'

        mask_split_edges(data, test_key, self.num_test, None)
        remaining_dict = {
            k: (~v).nonzero().flatten()
            for k, v in getattr(data, f'{test_key}_dict').items()
        }
        mask_split_edges(data, val_key, self.num_val, remaining_dict)

        for store in data.edge_stores:
            edge_type = store._key
            eval_mask = torch.logical_or(
                getattr(data, f'{test_key}_dict')[edge_type],
                getattr(data, f'{val_key}_dict')[edge_type]
            )
            train_mask = ~eval_mask
            setattr(store, train_key, train_mask)

        return data


def mask_split_edges(
    data: HeteroData,
    mask_key: str = 'train_mask',
    num_split: Union[int, float] = 0.1,
    remaining_dict: Dict[EdgeType, Tensor] = None,
) -> HeteroData:
    for store in data.edge_stores:
        edge_type = store._key
        mask = torch.zeros(store.num_edges, dtype=torch.bool)
        remaining = torch.arange(store.num_edges) if remaining_dict is None \
            else remaining_dict[edge_type]

        num_split_edges = round(store.num_edges * num_split)
        mask[torch.randperm(len(remaining))[:num_split_edges]] = True
        setattr(store, mask_key, mask)

    return data
