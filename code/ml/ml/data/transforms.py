from typing import Union, Optional, List

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


def sort_edges(
        node_count: int,
        edge_index: torch.Tensor,
        edge_attr: Optional[Union[Tensor, List[Tensor]]] = None
) -> Tensor:
    perm = torch.argsort(edge_index[0, :] * node_count + edge_index[1, :], dim=0, descending=False)
    edge_index = edge_index[:, perm]

    if edge_attr is not None and isinstance(edge_attr, Tensor):
        edge_attr = edge_attr[perm]
    elif edge_attr is not None:
        edge_attr = [e[perm] for e in edge_attr]

    return edge_index, edge_attr


class SortEdges(BaseTransform):
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            # Calculate num nodes
            if isinstance(data, HeteroData):
                src, rel, dst = store._key
                num_nodes = max(data[src].num_nodes, data[dst].num_nodes)
            else:
                num_nodes = data.num_nodes

            # Aggregate edge features
            keys, values = [], []
            for key, value in store.items():
                if key == 'edge_index':
                    continue

                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            # Sorted edge index
            store.edge_index, values = sort_edges(
                num_nodes,
                store.edge_index,
                values,
            )

            for key, value in zip(keys, values):
                store[key] = value

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
