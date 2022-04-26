import copy
from pathlib import Path
from typing import Dict, Union, Optional, List

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.loader.utils import edge_type_to_str, index_select
from torch_geometric.typing import NodeType, EdgeType, OptTensor
from torch_sparse import SparseTensor

from shared import get_logger

logger = get_logger(Path(__file__).stem)

NodeCountDict = Dict[NodeType, int]

NAN_TIMESTAMP = -1


def extract_attribute(
        data: HeteroData, key: str, edge_attr=False, warn=True, error=True
) -> Optional[Dict[Union[NodeType, EdgeType], Union[Tensor, np.ndarray]]]:
    if key not in data.keys:
        if warn:
            logger.warning(f"{key} not in data.keys")
            return None
        elif error:
            raise KeyError(f"{key} not in data.keys")

    stores = data.node_stores if not edge_attr else data.edge_stores

    is_numpy = False
    dtype = None
    has_attr = False
    for store in stores:
        if key in store.keys() and isinstance(store, EdgeStorage if edge_attr else NodeStorage):
            has_attr = True
            is_numpy = isinstance(store[key], np.ndarray)
            dtype = store[key].dtype

    if not has_attr:
        if warn:
            logger.warning(f"{key} not in any {'edge' if edge_attr else 'node'} store")
            return None
        elif error:
            raise KeyError(f"{key} not in any {'edge' if edge_attr else 'node'} store")

    output = {}
    for store in stores:
        entity_type = store._key

        if key in store.keys():
            output[entity_type] = store[key]
        else:
            if warn:
                logger.warning(f"{key} not in {entity_type} store")

            if is_numpy:
                output[entity_type] = np.full([store.num_nodes], -1, dtype=dtype)
            else:
                output[entity_type] = torch.full([store.num_nodes], -1, dtype=dtype)

    return output


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor) -> NodeStorage:
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif store.is_node_attr(key):
            index = index.to(value.device)
            out_store[key] = index_select(value, index, dim=0)

    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: Tensor,
                       perm: OptTensor = None) -> EdgeStorage:
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = torch.stack([row, col], dim=0)
            out_store.edge_index = edge_index.to(value.device)

        elif key == 'adj_t':
            # NOTE: We expect `(row, col)` to be sorted by `col` (CSC layout).
            row = row.to(value.device())
            col = col.to(value.device())
            edge_attr = value.storage.value()
            if edge_attr is not None:
                index = index.to(edge_attr.device)
                edge_attr = edge_attr[index]
            sparse_sizes = store.size()[::-1]
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=True)

        elif store.is_edge_attr(key):
            if perm is None:
                index = index.to(value.device)
                out_store[key] = index_select(value, index, dim=0)
            else:
                perm = perm.to(value.device)
                index = index.to(value.device)
                out_store[key] = index_select(value, perm[index], dim=0)

    return store


def filter_hetero_data(
        data: HeteroData,
        node_dict: Dict[str, Tensor],
        row_dict: Dict[str, Tensor],
        col_dict: Dict[str, Tensor],
        edge_dict: Dict[str, Tensor],
        perm_dict: Dict[str, OptTensor],
) -> HeteroData:
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(data)

    for node_type in data.node_types:
        filter_node_store_(data[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in data.edge_types:
        edge_type_str = edge_type_to_str(edge_type)
        filter_edge_store_(data[edge_type], out[edge_type],
                           row_dict[edge_type_str], col_dict[edge_type_str],
                           edge_dict[edge_type_str], perm_dict[edge_type_str])

    return out


def graph_clean_keys(data: Union[HeteroData, Data], key_whitelist: List[str]) -> Union[HeteroData, Data]:
    output = copy.copy(data)
    for store in output.stores:
        keys = [key for key in store.keys() if key not in key_whitelist]
        for key in keys:
            del store[key]

    return output