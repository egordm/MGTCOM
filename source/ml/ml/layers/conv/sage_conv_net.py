from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, SAGEConv, HeteroConv
from torch_geometric.typing import Metadata, NodeType
import torch.nn.functional as F

from ml.layers.conv.base import HeteroConvLayer
from ml.utils import HParams, dict_mapv


@dataclass
class SAGEConvNetParams(HParams):
    hidden_dim: Optional[int] = None

    num_layers: int = 2


class SAGEConvNet(HeteroConvLayer):
    def __init__(
            self,
            metadata: Metadata,
            repr_dim: int,
            num_layers: int = 2,
            hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.hidden_dim = self.hparams.hidden_dim or repr_dim
        node_types, edge_types = metadata

        self.convs = torch.nn.ModuleList([
            HeteroConv({
                edge_type: SAGEConv(
                    (-1, -1),
                    out_channels=hidden_dim if i < num_layers - 1 else self.repr_dim
                )
                for edge_type in edge_types
            })
            for i in range(num_layers)
        ])

    def convolve(self, data: HeteroData, X_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        # Apply convolutions
        Z_dict = X_dict
        for conv in self.convs:
            Z_dict = conv(Z_dict, data.edge_index_dict)
            Z_dict = dict_mapv(Z_dict, lambda z: F.leaky_relu(z))

        return self._process_batch(data, Z_dict)
