from typing import Dict

import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeType, EdgeType


class TypedHomogenousTransform(BaseTransform):
    def __call__(self, data: HeteroData) -> Data:
        node_type_to_idx = {
            node_type: idx
            for idx, node_type in enumerate(data.node_types)
        }
        edge_type_to_idx = {
            edge_type: idx
            for idx, edge_type in enumerate(data.edge_types)
        }

        output = Data()

        attr_node_perm = torch.zeros(data.num_nodes, dtype=torch.int64)
        attr_node_type = torch.zeros(data.num_nodes, dtype=torch.int64)
        node_offsets: Dict[NodeType, int] = {}
        offset = 0
        for store in data.node_stores:
            node_type = store._key
            attr_node_perm[offset:offset + store.num_nodes] = torch.arange(0, store.num_nodes, dtype=torch.int64)
            attr_node_type[offset:offset + store.num_nodes] = node_type_to_idx[node_type]
            node_offsets[node_type] = offset
            offset += store.num_nodes

        output.x = attr_node_perm
        output.node_type = attr_node_type
        output.node_offsets = node_offsets

        edges = torch.zeros((2, data.num_edges), dtype=torch.int64)
        attr_edge_perm = torch.zeros(data.num_edges, dtype=torch.int64)
        attr_edge_type = torch.zeros(data.num_edges, dtype=torch.int64)
        edge_offsets: Dict[EdgeType, int] = {}
        offset = 0
        for store in data.edge_stores:
            edge_type = store._key
            (src, _, dst) = edge_type
            edges[:, offset:offset + store.num_edges] = store.edge_index
            edges[0, offset:offset + store.num_edges] += node_offsets[src]
            edges[1, offset:offset + store.num_edges] += node_offsets[dst]
            attr_edge_perm[offset:offset + store.num_edges] = torch.arange(0, store.num_edges, dtype=torch.int64)
            attr_edge_type[offset:offset + store.num_edges] = edge_type_to_idx[edge_type]
            edge_offsets[edge_type] = offset
            offset += store.num_edges

        output.edge_index = edges
        output.edge_perm = attr_edge_perm
        output.edge_type = attr_edge_type
        output.edge_offsets = edge_offsets

        output.node_type_map = node_type_to_idx
        output.edge_type_map = edge_type_to_idx

        return output
