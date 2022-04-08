import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.typing import Metadata


class EmbeddingModule(torch.nn.Module):
    def __init__(self, metadata: Metadata, repr_dim: int = 32):
        super().__init__()
        self.metadata = metadata
        self.repr_dim = repr_dim


class HGTEmbeddingModule(EmbeddingModule):
    def __init__(
            self, metadata: Metadata,
            repr_dim: int = 32, hidden_dim: int = None,
            num_layers: int = 2, num_heads: int = 2,
            group: str = 'mean'
    ):
        super().__init__(metadata, repr_dim)
        self.hidden_dim = hidden_dim or repr_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.group = group

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, self.hidden_dim)

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim
            out_dim = self.repr_dim if i == self.num_layers - 1 else self.hidden_dim
            conv = HGTConv(in_dim, out_dim, metadata, self.num_heads, group=self.group)
            self.convs.append(conv)

    def forward(self, data: HeteroData):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in data.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        # Extract only layer 0 nodes
        for node_type in data.node_types:
            batch_size = data[node_type].batch_size
            x_dict[node_type] = x_dict[node_type][:batch_size, :]

        return x_dict
