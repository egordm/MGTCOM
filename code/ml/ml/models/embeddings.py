import torch.nn
from torch.nn import ReLU
from torch_geometric.data import HeteroData
from torch_geometric.nn import Sequential, to_hetero
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
