from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.typing import NodeType, Metadata

from ml.layers.embedding import HeteroNodeEmbedding
from ml.layers.hgt_cov_net import HGTConvNet, HGTConvNetParams

from ml.layers.sage_conv_net import SAGEConvNetParams, SAGEConvNet
from ml.utils import HParams


@dataclass
class HybridConvNetParams(HParams):
    repr_dim: int = 32
    hidden_dim: Optional[int] = None

    hgt_params: Optional[HGTConvNetParams] = None
    sage_params: Optional[SAGEConvNetParams] = None


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

        if self.hparams.hgt_params:
            self.conv = HGTConvNet(
                self.repr_dim,
                metadata,
                hparams=self.hparams.hgt_params,
            )
        elif self.hparams.sage_params:
            self.conv = SAGEConvNet(
                self.repr_dim,
                metadata,
                hparams=self.hparams.sage_params,
            )
        else:
            self.conv = None

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

        # Apply convolutions
        if self.conv:
            Z_dict = self.conv(data, X_dict)
        else:
            Z_dict = {
                node_type: X_dict[node_type][:batch_size]
                for node_type, batch_size in data.batch_size_dict.items()
            }

        return Z_dict
