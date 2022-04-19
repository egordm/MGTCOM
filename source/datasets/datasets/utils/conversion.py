from collections import defaultdict
from typing import Dict, Tuple

import igraph
import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType, EdgeType


def igraph_from_hetero(
        data: HeteroData,
        node_attrs: Dict[str, Dict[NodeType, Tensor]] = None,
        edge_attrs: Dict[str, Dict[EdgeType, Tensor]] = None,
) -> Tuple[igraph.Graph, Dict[NodeType, int], Dict[EdgeType, int], Dict[NodeType, int]]:
    """Convert a heterograph to an igraph graph."""
    node_type_to_idx = {
        node_type: idx
        for idx, node_type in enumerate(data.node_types)
    }
    edge_type_to_idx = {
        edge_type: idx
        for idx, edge_type in enumerate(data.edge_types)
    }

    g = igraph.Graph()
    g.add_vertices(data.num_nodes)

    attr_node_perm = np.zeros(data.num_nodes, dtype=np.int64)
    attr_node_type = np.zeros(data.num_nodes, dtype=np.int64)
    node_perms: Dict[NodeType, Tensor] = {}
    offsets: Dict[NodeType, int] = {}
    offset = 0
    for store in data.node_stores:
        node_type = store._key
        attr_node_perm[offset:offset + store.num_nodes] = np.arange(0, store.num_nodes, dtype=np.int64)
        attr_node_type[offset:offset + store.num_nodes] = node_type_to_idx[node_type]
        offsets[node_type] = offset
        offset += store.num_nodes

    g.vs["id"] = attr_node_perm
    g.vs["type"] = attr_node_type

    for attr_name, attr_data in (node_attrs or {}).items():
        attr_items = []
        for store in data.node_stores:
            attr = attr_data[store._key]
            attr_items.append(attr.numpy() if torch.is_tensor(attr) else attr)

        g.vs[attr_name] = np.concatenate(attr_items, axis=0)

    edges = np.zeros((data.num_edges, 2), dtype=np.int64)
    attr_edge_perm = np.zeros(data.num_edges, dtype=np.int64)
    attr_edge_type = np.zeros(data.num_edges, dtype=np.int64)
    offset = 0
    for store in data.edge_stores:
        edge_type = store._key
        (src, _, dst) = edge_type
        edges[offset:offset + store.num_edges, :] = store.edge_index.t().numpy()
        edges[offset:offset + store.num_edges, 0] += offsets[src]
        edges[offset:offset + store.num_edges, 1] += offsets[dst]
        attr_edge_perm[offset:offset + store.num_edges] = np.arange(0, store.num_edges, dtype=np.int64)
        attr_edge_type[offset:offset + store.num_edges] = edge_type_to_idx[edge_type]
        offset += store.num_edges

    g.add_edges(edges)
    g.es["id"] = attr_edge_perm
    g.es["type"] = attr_edge_type

    for attr_name, attr_data in (edge_attrs or {}).items():
        attr_items = []
        for store in data.edge_stores:
            attr = attr_data[store._key]
            attr_items.append(attr.numpy() if torch.is_tensor(attr) else attr)

        g.es[attr_name] = np.concatenate(attr_items, axis=0)

    return g, node_type_to_idx, edge_type_to_idx, offsets
