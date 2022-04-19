from typing import Union, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomNodeSplit as RandomNodeSplitTG
from torch_geometric.typing import NodeType


class RandomNodeSplit(RandomNodeSplitTG):

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        result = super().__call__(data)

        mask_edges_by_node_mask(result, result.train_mask_dict, 'train_mask', inplace=True)
        mask_edges_by_node_mask(result, result.val_mask_dict, 'val_mask', inplace=True)
        mask_edges_by_node_mask(result, result.test_mask_dict, 'test_mask', inplace=True)

        return result


def mask_edges_by_node_mask(
        data: HeteroData,
        node_mask_dict: Dict[NodeType, Tensor],
        mask_name: str, inplace=True
) -> HeteroData:
    out = data if inplace else data.clone()

    for store in out.edge_stores:
        src, _, dst = store._key
        mask_src = node_mask_dict[src][store.edge_index[0, :]]
        mask_dst = node_mask_dict[dst][store.edge_index[1, :]]
        mask = torch.logical_or(mask_src, mask_dst)
        store[mask_name] = mask

    return data
