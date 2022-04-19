from typing import Dict, Union, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import NodeType


class NodeEmbedding(torch.nn.Module):
    def __init__(self, num_nodes: int, repr_dim: int) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.num_nodes = num_nodes

        self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=repr_dim)

    def forward(self, node_idx: Tensor) -> Tensor:
        Z = torch.zeros(len(node_idx), self.repr_dim)

        existing_mask = (node_idx < self.num_nodes).nonzero()
        Z[existing_mask] = self.embedding(node_idx[existing_mask])
        return Z


HeteroNodeEmbeddingInput = Tuple[Dict[NodeType, Tensor], Dict[NodeType, Tensor]]


class HeteroNodeEmbedding(torch.nn.Module):
    def __init__(
            self,
            num_nodes_dict: Dict[NodeType, int],
            repr_dim: Union[int, Dict[NodeType, int]],
    ):
        super().__init__()
        self.num_nodes_dict = num_nodes_dict
        self.repr_dim_dict = repr_dim if isinstance(repr_dim, dict) \
            else {node_type: repr_dim for node_type in num_nodes_dict}
        self.repr_dim = next(iter(self.repr_dim_dict.values()), 0)

        self.embedding_dict = torch.nn.ModuleDict({
            node_type: NodeEmbedding(num_nodes, self.repr_dim_dict[node_type])
            for node_type, num_nodes in num_nodes_dict.items()
        })

    def forward(self, node_idx_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        return {
            node_type: self.embedding_dict[node_type](node_idx)
            for node_type, node_idx in node_idx_dict.items()
        }
