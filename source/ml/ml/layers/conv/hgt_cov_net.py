from typing import Optional, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.typing import Metadata, NodeType

from ml.layers.conv.base import HeteroConvLayer


class HGTConvNet(HeteroConvLayer):
    def __init__(
            self,
            metadata: Metadata,
            repr_dim: int,
            num_layers: int = 2,
            heads: int = 2,
            hidden_dim: Optional[int] = None,
            input_dim: Optional[int] = None,
            group: str = 'mean',
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.input_dim = input_dim or hidden_dim or repr_dim
        self.hidden_dim = hidden_dim or repr_dim
        self.num_layers = num_layers
        self.heads = heads
        self.group = group

        self.convs = torch.nn.ModuleList([
            HGTConv(
                metadata=metadata,
                in_channels=self.input_dim if i == 0 else self.hidden_dim,
                out_channels=self.hidden_dim if i < self.num_layers - 1 else self.repr_dim,
                heads=heads,
                group=group
            )
            for i in range(self.num_layers)
        ])

    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor] = None) -> Dict[NodeType, Tensor]:
        # Apply convolutions
        Z_dict = X_dict
        for conv in self.convs:
            Z_dict = conv(Z_dict, data.edge_index_dict)

        return Z_dict
