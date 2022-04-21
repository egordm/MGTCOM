import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear

from ml.layers.conv.hgt_cov_net import HGTConvNet
from ml.layers.embedding import NodeEmbedding

path = osp.join(osp.dirname(osp.realpath(__file__)), './data/DBLP')
dataset = DBLP(path)
data = dataset[0]
print(data)

# We initialize conference node features with a single feature.
data['conference'].x = torch.ones(data['conference'].num_nodes, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.conf_emb = NodeEmbedding(data['conference'].num_nodes, hidden_channels)

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.conv = HGTConvNet(hidden_channels, data.metadata())

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data: HeteroData):
        for node_type in data.node_types:
            data[node_type].batch_size = data[node_type].num_nodes

        x_dict = {
            node_type: x
            for node_type, x in data.x_dict.items()
        }
        x_dict['conference'] = self.conf_emb(torch.arange(data['conference'].num_nodes, device=device))

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        z_dict = self.conv(data, x_dict)

        return self.lin(z_dict['author'])


model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, '
          f'Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}')