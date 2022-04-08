from typing import Tuple, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


def extract_unique_nodes(data: HeteroData) -> Tuple[Dict[NodeType, Tensor], HeteroData]:
    offset = 0
    nodes_dict = {}
    for store in data.node_stores:
        nodes, perm = torch.unique(store.x, return_inverse=True)
        nodes_dict[store._key] = nodes
        store.perm = perm + offset
        offset += len(nodes)

    return nodes_dict, data
