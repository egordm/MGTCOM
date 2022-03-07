from typing import List, Tuple, Union
import random

import pytorch_lightning as pl
import torch
import torchmetrics
import pandas as pd
import numpy as np
import torch_geometric.nn as tg_nn
from tch_geometric.data import to_csr, to_csc
from tch_geometric.transforms import WeightedEdgeSampler
from torch_geometric.transforms import ToUndirected
import tch_geometric as thg
from torch_geometric.loader.utils import filter_data
from torch_geometric.data import Data
from torch_geometric.loader.base import BaseDataLoader
from torch_scatter.utils import broadcast
from torch_scatter import scatter_sum, scatter_mean
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from experiments import ClusteringModule, ClusterCohesionLoss, NegativeEntropyRegularizer, cosine_cdist, save_projector, \
    EmbeddingModule
from ml import SortEdges
from shared.constants import TMP_PATH, BENCHMARKS_RESULTS
from shared.graph import CommunityAssignment
import ml
import experiments

experiment_name = 'pyg-pinsage'
node_type = 'Character'
repr_dim = 32
EPS = 1e-15
recluster = False
save_path = TMP_PATH.joinpath('pyg-pinsage')
save = False

dataset = ml.StarWarsHomogenous()
data = dataset[0]
data = Data(data.node_stores[0].x, data.edge_stores[0].edge_index)
data = ToUndirected()(data)
data = SortEdges()(data)
G = dataset.G
G.to_undirected()



# Node Importance Definition
degree_avg = np.mean(dataset.G.degree())
row_ptrs, col_indices, perm, size = to_csr(data)
# row_ptrs, col_indices, perm = thg.data.to_csr(data.edge_index, data.num_nodes)
neighbor_weight = torch.ones(len(data.edge_index[0, :]))

walks_per_node = int(np.ceil(degree_avg))
walk_length = 5
start = torch.arange(data.num_nodes, dtype=torch.long)
walks = thg.native.random_walk(row_ptrs, col_indices, start.repeat(walks_per_node), walk_length, p=1.0, q=1.0)

walk_edges = []
for i in range(walk_length - 1):
    walk_edges.append(walks[:, i:i + 2])
walk_edges = torch.cat(walk_edges, dim=0)
walk_edges, walk_edge_counts = torch.unique(walk_edges, dim=0, return_counts=True, sorted=True)

left = data.edge_index[0, :] * data.num_nodes + data.edge_index[1, :]
right = walk_edges[:, 0] * data.num_nodes + walk_edges[:, 1]
match = torch.searchsorted(left, right, right=True)

walk_edge_counts = walk_edge_counts[match != 0]
match = match[match != 0] - 1
neighbor_weight[match] += walk_edge_counts
data.edge_stores[0].weight = neighbor_weight

# Build pos and neg samples
rows, cols = data.edge_index
csr = csr_matrix(
    (torch.ones(data.edge_index.shape[1], dtype=torch.int32).numpy(), (rows.numpy(), cols.numpy())),
    shape=(data.num_nodes, data.num_nodes),
)
neg_neighbors = [
    list(set(range(data.num_nodes)).difference(set(csr[i, :].indices)))
    for i in range(data.num_nodes)
]


def neg_sample(batch: torch.Tensor, num_neg_samples: int = 1) -> torch.Tensor:
    result = torch.tensor([
        random.choices(neg_neighbors[i], k=num_neg_samples)
        for i in batch[:, 0]
    ], dtype=torch.long)
    return result


col_ptrs, row_indices, csc_perm, size = to_csc(data)

def cooler_neg_sample(batch: torch.Tensor, num_neg_samples: int = 1) -> torch.Tensor:
    inputs = batch[:, 0]
    neg_idx, errors = thg.native.negative_sample_neighbors(col_ptrs, row_indices, data.size(), inputs, num_neg_samples, 10)

    return neg_idx[1, :].reshape([len(inputs), num_neg_samples])



repeat_count = 2
num_neg_samples = 3
pos_idx = data.edge_index.t().repeat(repeat_count, 1)
# neg_idx = negative_sampling(data.edge_index).t()
neg_idx = neg_sample(pos_idx, num_neg_samples=num_neg_samples)
# neg_idx = cooler_neg_sample(pos_idx, num_neg_samples=num_neg_samples)

data_idx = torch.cat([pos_idx, neg_idx], dim=1)


# Build dataloader
class WeightedNeighborLoader:
    def __init__(self, data: Data, num_neighbors: List[int]):
        self.num_neighbors = num_neighbors
        self.col_ptrs, self.row_indices, self.perm, self.size = to_csc(data)
        self.weights = data.edge_stores[0].weight[self.perm].double()
        # super(NeighborLoader, self).__init__(data, batch_size, shuffle)

    def __call__(self, indices: List[int]):
        index = torch.tensor(indices)
        nodes, rows, cols, edge_index, layer_offsets = thg.native.neighbor_sampling_homogenous(
            self.col_ptrs, self.row_indices, index, self.num_neighbors, sampler=WeightedEdgeSampler(self.weights)
        )
        return nodes, rows, cols, edge_index, len(indices), layer_offsets


class NodesLoader(BaseDataLoader):
    def __init__(self, data: Data, inputs: torch.Tensor, num_neighbors: List[int], **kwargs):
        self.data = data
        self.neighbor_sampler = WeightedNeighborLoader(data, num_neighbors)

        super().__init__(
            inputs.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long).tolist()
        return self.neighbor_sampler(indices)

    def transform_fn(self, out):
        nodes, rows, cols, edge_index, batch_size, layer_offsets = out
        data = filter_data(self.data, nodes, rows, cols, edge_index, self.neighbor_sampler.perm)
        data.batch_size = batch_size
        return data, layer_offsets


class SamplesLoader(NodesLoader):
    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long).view(-1).tolist()
        return self.neighbor_sampler(indices)


num_neighbors = [4, 4]
batch_size = 8
nodes_loader = NodesLoader(
    data, torch.arange(data.num_nodes), num_neighbors,
    batch_size=batch_size, num_workers=0
)
data_loader = SamplesLoader(
    data, data_idx, num_neighbors,
    batch_size=batch_size, num_workers=0, shuffle=True
)
batch, batch_layer_offsets = next(iter(data_loader))

# Build model
use_cosine = False
use_centers = True
initialize = None  # 'louvain'
# initialize = 'louvain'
n_clusters = 5

n_epochs = 30  # 5 # 30
comm_epoch = 10  # 2 #10


# n_epochs = 50  # 5 # 30
# comm_epoch = 30  # 2 #10


# n_epochs = 5 # 30
# comm_epoch = 2 #10

def initial_clustering(embeddings: torch.Tensor):
    clustering = G.community_multilevel(return_levels=True)[0]
    assignment = torch.tensor(clustering.membership)
    cluster_count = len(clustering)
    assigned_count = torch.zeros(cluster_count, dtype=torch.long) \
        .scatter_add(0, assignment, torch.ones_like(assignment, dtype=torch.long))
    c = torch.zeros(cluster_count, repr_dim, dtype=torch.float).index_add_(0, assignment, embeddings)
    c = c / assigned_count.unsqueeze(1)
    return c


# Build model
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

    def forward_old(self, x: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index, x, w, dim_size=len(x))

        h_n = self.lin_l(agg)
        h_v = self.lin_r(x)
        out = h_n + h_v

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        z_n = self.relu(self.lin_l(x))
        agg = self.propagate(edge_index, z_n, w, dim_size=len(x))
        out = self.relu(self.lin_r(torch.cat([x, agg], dim=1)))

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class GraphSAGEHeteroModuleNew(torch.nn.Module):
    def __init__(self, repr_dim: int = 32, n_layers: int = 2, normalize=False) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.n_layers = n_layers

        self.conv1 = PinSAGELayer((32, repr_dim), repr_dim, normalize)
        self.conv2 = PinSAGELayer((32, repr_dim), repr_dim, normalize)
        self.activation = torch.nn.ReLU()

    def forward(self, batch: Data, layer_offsets: List[Tuple[int, int]]) -> torch.Tensor:
        x = batch.x
        x = self.conv1(x, batch.edge_index, batch.weight)
        x = self.activation(x)
        x = self.conv2(x, batch.edge_index, batch.weight)
        return x[:batch.batch_size]


class GraphSAGEHeteroModuleReplica(torch.nn.Module):
    def __init__(
            self, repr_dim: int = 32, n_layers: int = 2, normalize=False
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.n_layers = n_layers
        self.normalize = normalize

        self.lin_l1 = torch.nn.Linear(32, repr_dim, bias=True)
        self.lin_l2 = torch.nn.Linear(32, repr_dim, bias=True)
        self.lin_r1 = torch.nn.Linear(32, repr_dim, bias=False)
        self.lin_r2 = torch.nn.Linear(32, repr_dim, bias=False)

        self.lin_l = [self.lin_l1, self.lin_l2]
        self.lin_r = [self.lin_r1, self.lin_r2]
        self.activation = torch.nn.ReLU(inplace=True)

    def propagate(self, edge_index, x):
        gather = x.index_select(0, edge_index[0, :])
        agg = scatter_mean(gather, edge_index[1, :], dim=0, dim_size=x.shape[0])
        return agg

    def conv(self, i, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        h_n = self.lin_l[i](agg)
        h_v = self.lin_r[i](x)

        h = h_n + h_v
        return h

    def forward(self, batch: Data, *args) -> torch.Tensor:
        x = batch.x
        x = self.conv(0, x, batch.edge_index)
        x = self.activation(x)
        x = self.conv(1, x, batch.edge_index)
        return x[:batch.batch_size]


embedding_module = GraphSAGEHeteroModuleNew(
    repr_dim, n_layers=2, normalize=False
)


# embedding_module = GraphSAGEHeteroModuleReplica(
#     repr_dim, n_layers=2, normalize=False
# )

# embedding_module = GraphSAGEHeteroConvModule2(
#     repr_dim, n_layers=2, normalize=False
# )

# embedding_module = experiments.GraphSAGEHeteroModule(
#     repr_dim, n_layers=2, normalize=False
# )


class MainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)

    def forward(self, batch, optimize_comms=False):
        batch, layer_offsets = batch
        emb = embedding_module(batch, layer_offsets)
        neg_pos_emb = emb.view(-1, num_neg_samples + 2, repr_dim)
        ctr_emb = neg_pos_emb[:, 0, :].unsqueeze(1)
        pos_emb = neg_pos_emb[:, 1, :].unsqueeze(1)
        neg_emb = neg_pos_emb[:, 2:, :]

        # Max Margin loss
        pos_d = torch.bmm(ctr_emb, pos_emb.transpose(1, 2)).view(-1)
        neg_d = torch.bmm(ctr_emb, neg_emb.transpose(1, 2)).max(dim=2).values.view(-1)
        mm_loss = torch.clip(neg_d - pos_d + 1, min=0).mean()
        loss = mm_loss

        if optimize_comms:
            # Clustering Max Margin loss
            current_batch_size = neg_pos_emb.shape[0]
            c_emb = self.centroids.weight.clone().unsqueeze(0).transpose(1, 2).repeat(current_batch_size, 1, 1)
            c_ctr_q = torch.softmax(torch.bmm(ctr_emb, c_emb), dim=-1)
            c_pos_q = torch.softmax(torch.bmm(pos_emb, c_emb), dim=-1)
            c_neg_q = torch.softmax(torch.bmm(neg_emb, c_emb), dim=-1)

            c_pos_d = torch.bmm(c_ctr_q, c_pos_q.transpose(1, 2)).view(-1)
            c_neg_d = torch.bmm(c_ctr_q, c_neg_q.transpose(1, 2)).max(dim=2).values.view(-1)
            c_mm_loss = torch.clip(c_neg_d - c_pos_d + 1, min=0).mean()
            loss = mm_loss + c_mm_loss

        # c_emb.unsqueeze(0).transpose(1, 2)
        return loss


model = MainModel()
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


def train(epoch):
    model.train()
    total_loss = 0
    optimize_comms = epoch >= comm_epoch
    for batch in data_loader:
        optimizer.zero_grad()

        loss = model(batch, optimize_comms)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def get_embeddings():
    model.eval()
    embs = []
    for batch, layer_offsets in nodes_loader:
        emb = embedding_module(batch, layer_offsets)
        embs.append(emb)

    return torch.cat(embs, dim=0)


for epoch in range(1, n_epochs):
    if epoch == comm_epoch and initialize is not None:
        embeddings = get_embeddings()
        c = initial_clustering(embeddings)
        model.centroids = torch.nn.Embedding.from_pretrained(c, freeze=False)

    loss = train(epoch)
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #

if save:
    save_path.mkdir(exist_ok=True, parents=True)
    torch.save(embedding_module.state_dict(), save_path.joinpath('embeddings.pt'))
    torch.save(model.centroids.state_dict(), save_path.joinpath('centroids.pt'))
    print(f'Saved model to {save_path}')
u = 0
# aaaa

from shared.constants import BENCHMARKS_RESULTS
import faiss
import pandas as pd

save_path = BENCHMARKS_RESULTS.joinpath('analysis', experiment_name)
save_path.mkdir(parents=True, exist_ok=True)

embeddings = get_embeddings().detach()

if use_centers:
    print('Reusing trained centers')
    centers = model.centroids.weight.detach()
    q = torch.softmax(torch.mm(embeddings, centers.transpose(1, 0)), dim=-1)
    I = q.argmax(dim=-1)
    I = I.numpy()
else:
    if use_cosine:
        print('Normalize for cosine similarity')
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    print('Searching for cluster centers using K-means')
    kmeans = faiss.Kmeans(embeddings.shape[1], k=n_clusters, niter=20, verbose=True, nredo=10)
    kmeans.train(embeddings.numpy())
    D, I = kmeans.index.search(embeddings.numpy(), 1)

if save:
    save_projector("Star Wars Positional", embeddings, pd.DataFrame({
        'label': G.vs['label'],
        'cluster': I
    }))

from shared.graph import CommunityAssignment

labeling = pd.Series(I.squeeze(), index=dataset.node_mapping(), name="cid")
labeling.index.name = "nid"
comlist = CommunityAssignment(labeling)
comlist.save_comlist(save_path.joinpath('schema.comlist'))

from datasets.scripts import export_to_visualization
from shared.graph import DataGraph
from benchmarks.evaluation import get_metric_list
from shared.schema import GraphSchema, DatasetSchema

export_to_visualization.run(
    export_to_visualization.Args(
        dataset='star-wars',
        version='base',
        run_paths=[str(save_path)]
    )
)

DATASET = DatasetSchema.load_schema('star-wars')
schema = GraphSchema.from_dataset(DATASET)
# G = DataGraph.from_schema(schema)
# G.to_undirected()

metrics = get_metric_list(ground_truth=False, overlapping=False)
results = pd.DataFrame([
    {
        'metric': metric_cls.metric_name(),
        'value': metric_cls.calculate(G, comlist)
    }
    for metric_cls in metrics]
)
print(results)
