from typing import Callable, Tuple, List

import numpy as np
import torch
from tch_geometric.loader import CustomLoader
from tch_geometric.transforms import NegativeSamplerTransform, NeighborSamplerTransform
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_scatter import scatter_sum, scatter_max
import torch.nn.functional as F

import ml
from experiments import HingeLoss, NegativeEntropyRegularizer
from ml import SortEdges

dataset = ml.StarWarsHomogenous()
data = dataset[0]
data = Data(data.node_stores[0].x, data.edge_stores[0].edge_index)
data = ToUndirected()(data)
data = SortEdges()(data)
G = dataset.G
G.to_undirected()

repr_dim = 32
n_epochs = 20 # 10
n_comm_epochs = 10
n_clusters = 5
batch_size = 10


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]


dataset = CustomDataset(data.edge_index)

neg_sampler = NegativeSamplerTransform(data, 3, 5)
neighbor_sampler = NeighborSamplerTransform(data, [4, 3])


class DataLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.neg_sampler = neg_sampler
        self.neighbor_sampler = neighbor_sampler

    def sample(self, inputs):
        batch_size = inputs.shape[1]
        ctr_idx = inputs[0, :]

        # Positive samples
        # TODO: use positive sampler instead
        pos_nodes = inputs.view(-1)
        pos_edges = (
            torch.arange(0, batch_size, dtype=torch.long),
            torch.arange(batch_size, batch_size * 2, dtype=torch.long)
        )

        # Negative samples
        neg_nodes, neg_edges, _ = self.neg_sampler(ctr_idx)
        neg_edges = (neg_edges[0], neg_edges[1] + len(pos_nodes))

        # Neighbor samples
        nodes, inverse_indices = torch.unique(
            torch.cat([pos_nodes, neg_nodes]),
            return_inverse=True,
        )
        samples = self.neighbor_sampler(nodes)
        samples.edge_weight = torch.ones(samples.edge_index.shape[1], dtype=torch.float)

        return samples, inverse_indices, pos_edges, neg_edges, batch_size


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
batch = next(iter(data_loader))
u = 0


class PinSAGELayer(torch.nn.Module):
    def __init__(self, in_channels: Tuple[int, int], out_channels: int, normalize: bool) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_u_channels, self.in_v_channels = in_channels
        self.normalize = normalize

        self.lin_l = torch.nn.Linear(self.in_v_channels, out_channels, bias=True)
        self.lin_r = torch.nn.Linear(self.in_u_channels + out_channels, out_channels, bias=True)
        self.relu = torch.nn.ReLU()

    def propagate(self, edge_index: torch.Tensor, x: torch.Tensor, w: torch.Tensor, dim_size: int):
        agg_w = scatter_sum(w, edge_index[1, :], dim_size=dim_size, dim=0).unsqueeze(1) + 1
        gather = x.index_select(0, edge_index[0, :]) * w.unsqueeze(1)
        agg = scatter_sum(gather, edge_index[1, :], dim=0, dim_size=dim_size) / agg_w
        return agg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        z_n = self.relu(self.lin_l(x))
        agg = self.propagate(edge_index, z_n, w, dim_size=len(x))
        out = self.relu(self.lin_r(torch.cat([x, agg], dim=1)))

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class EmbeddingModule(torch.nn.Module):
    def __init__(self, repr_dim: int = 32, n_layers: int = 2, normalize=False) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.n_layers = n_layers

        self.conv1 = PinSAGELayer((32, repr_dim), repr_dim, normalize)
        self.conv2 = PinSAGELayer((32, repr_dim), repr_dim, normalize)
        self.activation = torch.nn.ReLU()

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x
        x = self.conv1(x, batch.edge_index, batch.edge_weight)
        x = self.activation(x)
        x = self.conv2(x, batch.edge_index, batch.edge_weight)
        return x[:batch.batch_size]


embedding_module = EmbeddingModule(repr_dim, n_layers=2, normalize=False)


class MainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)
        self.posi_loss = HingeLoss()
        self.clus_loss = HingeLoss()
        self.neg_entropy = NegativeEntropyRegularizer()

    def forward(self, batch, optimize_comms=False):
        samples, inverse_indices, pos_edges, neg_edges, batch_size = batch
        emb = embedding_module(samples)

        posi_loss = self.posi_loss(emb, inverse_indices, pos_edges, neg_edges)
        loss = posi_loss

        if optimize_comms:
            # Clustering Max Margin loss
            c_emb = self.centroids.weight.clone().unsqueeze(0)
            c_emb_q = torch.softmax(torch.sum(emb.unsqueeze(1) * c_emb, dim=2), dim=1)

            clus_loss = self.clus_loss(c_emb_q, inverse_indices, pos_edges, neg_edges)
            ne = self.neg_entropy(c_emb_q)
            loss = posi_loss + clus_loss + ne * 0.01

        return loss


model = MainModel()
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


def train(epoch):
    model.train()
    total_loss = 0
    optimize_comms = epoch > n_epochs
    for batch in data_loader:
        optimizer.zero_grad()

        loss = model(batch, optimize_comms)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


for epoch in range(1, n_epochs + n_comm_epochs):
    # if epoch == comm_epoch and initialize is not None:
    #     embeddings = get_embeddings()
    #     c = initial_clustering(embeddings)
    #     model.centroids = torch.nn.Embedding.from_pretrained(c, freeze=False)

    loss = train(epoch)
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #