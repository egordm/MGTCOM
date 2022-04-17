from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.typing import NodeType, Metadata

from ml.layers.embedding import HeteroNodeEmbedding
from ml.utils import HParams


@dataclass
class HybridConvNetParams(HParams):
    repr_dim: int = 32
    hidden_dim: Optional[int] = None

    num_layers: int = 2
    num_heads: int = 2


class HybridConvNet(torch.nn.Module):
    def __init__(
            self,
            metadata: Metadata,
            embed_num_nodes: Dict[NodeType, int],
            hparams: HybridConvNetParams = None
    ) -> None:
        super().__init__()
        self.hparams = hparams or HybridConvNetParams()
        self.node_types, self.edge_types = metadata
        self.hidden_dim = self.hparams.hidden_dim or self.hparams.repr_dim
        self.repr_dim = self.hparams.repr_dim

        self.node_types_embed = set(embed_num_nodes.keys())
        self.embedding = HeteroNodeEmbedding(embed_num_nodes, self.hidden_dim)

        self.lin_dict = torch.nn.ModuleDict({
            node_type: Linear(-1, self.hidden_dim)
            for node_type in self.node_types if node_type not in self.node_types_embed
        })

        self.dropout = torch.nn.Dropout(p=0.5)

        self.convs = torch.nn.ModuleList()
        for i in range(self.hparams.num_layers):
            out_dim = self.hidden_dim if i < self.hparams.num_layers - 1 else self.repr_dim
            self.convs.extend([
                HGTConv(self.hidden_dim, out_dim, metadata, self.hparams.num_heads, group='mean')
            ])

    def forward(self, data: HeteroData) -> Dict[NodeType, Tensor]:
        # Process nodes with included features
        X_feat = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in data.x_dict.items() if node_type not in self.node_types_embed
        }

        # Process nodes with missing/embedded features
        embed_idx_dict = {
            node_type: idx
            for node_type, idx in data.node_idx_dict.items() if node_type in self.node_types_embed
        }
        # Apply dropout to ensure we don't rely too much embeddings
        X_embed = {
            node_type: self.dropout(x)
            for node_type, x in self.embedding(embed_idx_dict).items()
        }
        X_dict = {**X_feat, **X_embed}

        Z_dict = X_dict

        # Apply convolutions
        for conv in self.convs:
            Z_dict = conv(Z_dict, data.edge_index_dict)

        # Return node represenations
        return {
            node_type: Z_dict[node_type][:batch_size]
            for node_type, batch_size in data.batch_size_dict.items()
        }
