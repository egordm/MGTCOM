from typing import Dict, Union

import torch.nn
from torch.nn import ReLU
from torch_geometric.data import HeteroData
from torch_geometric.nn import Sequential, Linear, to_hetero
from torch_geometric.typing import Metadata, NodeType

from ml.layers.hgt_conv import HGTConv
from ml.layers.pinsage_conv import PinSAGEConv


class PinSAGEModule(torch.nn.Module):
    def __init__(self, metadata: Metadata, repr_dim: int = 32, normalize=True) -> None:
        super().__init__()
        self.repr_dim = repr_dim

        module = Sequential('x, edge_index', [
            (PinSAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (PinSAGEConv((-1, -1), repr_dim, normalize=normalize), 'x, edge_index -> x'),
        ])
        self.module = to_hetero(module, metadata, aggr='mean', debug=True)

    def forward(self, data: HeteroData) -> torch.Tensor:
        emb = self.module(data.x_dict, data.edge_index_dict)
        for node_type in data.node_types:
            batch_size = data[node_type].batch_size
            emb[node_type] = emb[node_type][:batch_size, :]

        return emb


class HGTModule(torch.nn.Module):
    def __init__(
            self,
            metadata: Metadata,
            in_channels: Union[int, Dict[NodeType, int]], out_channels: int, hidden_channels: int = None,
            num_heads=2, num_layers=2, group='mean',
            use_RTE=False, use_Lin=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or out_channels
        self.use_RTE = use_RTE
        self.use_Lin = use_Lin

        if use_Lin:
            self.lin_dict = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                self.lin_dict[node_type] = Linear(-1, self.hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            l_in_channels = self.in_channels if i == 0 and not use_Lin else self.hidden_channels
            l_out_channels = self.out_channels if i == num_layers - 1 else self.hidden_channels
            conv = HGTConv(l_in_channels, l_out_channels, metadata, num_heads, group=group, use_RTE=use_RTE)
            self.convs.append(conv)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        timedelta_dict = None
        if self.use_RTE:
            timedelta_dict = data.timedelta_dict

        if self.use_Lin:
            x_dict = {
                node_type: self.lin_dict[node_type](x).relu_()
                for node_type, x in x_dict.items()
            }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, timedelta_dict)

        for node_type in data.node_types:
            batch_size = data[node_type].batch_size
            x_dict[node_type] = x_dict[node_type][:batch_size, :]

        return x_dict
