from typing import Dict, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear
from torch_geometric.typing import NodeType, Metadata

from ml.layers.conv.base import HeteroConvLayer
from ml.layers.embedding import HeteroNodeEmbedding
from ml.utils import dict_mapv


class HybridConvNet(HeteroConvLayer):
    def __init__(
            self,
            metadata: Metadata,
            embed_num_nodes: Dict[NodeType, int],
            conv: Optional[HeteroConvLayer],
            hidden_dim: Optional[int] = None,
            repr_dim: Optional[int] = None,
            use_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.repr_dim = repr_dim or conv.repr_dim
        self.hidden_dim = hidden_dim or self.repr_dim
        self.use_dropout = use_dropout

        self.node_types_embed = set(embed_num_nodes.keys())
        self.embedding = HeteroNodeEmbedding(embed_num_nodes, self.hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.conv = conv
        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, self.hidden_dim)
            for node_type in self.node_types if node_type not in self.node_types_embed
        })

    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        # Process nodes with included features
        X_feat = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in X_dict.items() if node_type not in self.node_types_embed
        }

        # Process nodes with missing/embedded features
        embed_idx_dict = {
            node_type: idx
            for node_type, idx in data.node_idx_dict.items() if node_type in self.node_types_embed
        }
        X_embed = self.embedding(embed_idx_dict)
        if self.use_dropout:
            X_embed = dict_mapv(X_embed, lambda x: self.dropout(x))

        # Combine features and embedded features
        X_dict = {
            node_type: X_embed[node_type] if node_type in X_embed else X_feat[node_type]
            for node_type in self.node_types if node_type in X_feat.keys() | X_embed.keys()
        }

        # Convolve if applicable
        if self.conv is not None:
            Z_dict = self.conv(data, X_dict, return_raw=True)
        else:
            Z_dict = X_dict

        return Z_dict
