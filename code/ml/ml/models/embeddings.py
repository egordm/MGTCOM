import torch.nn
from torch.nn import ReLU
from torch_geometric.data import HeteroData
from torch_geometric.nn import Sequential, to_hetero, HGTConv, Linear
from torch_geometric.typing import Metadata

from ml.layers.pinsage import PinSAGEConv


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
    def __init__(self, metadata: Metadata, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        for node_type in data.node_types:
            batch_size = data[node_type].batch_size
            x_dict[node_type] = x_dict[node_type][:batch_size, :]

        return x_dict