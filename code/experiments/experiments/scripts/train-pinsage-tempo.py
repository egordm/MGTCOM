from typing import List, Tuple, Union
import random

import pytorch_lightning as pl
import torch
import torchmetrics
import pandas as pd
import numpy as np
import torch_geometric.nn as tg_nn
from torch_geometric.transforms import ToUndirected
import tch_geometric as thg
from torch_geometric.loader.utils import filter_data
from torch_geometric.data import Data
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.loader import NeighborLoader
from torch_scatter.utils import broadcast
from torch_scatter import scatter_sum, scatter_mean
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import itertools as it

from experiments import ClusteringModule, ClusterCohesionLoss, NegativeEntropyRegularizer, cosine_cdist, save_projector, \
    EmbeddingModule
from ml import SortEdges
from shared.constants import TMP_PATH, BENCHMARKS_RESULTS
from shared.graph import CommunityAssignment
import ml
import experiments

experiment_name = 'pyg-pinsage-tempo'
node_type = 'Character'
repr_dim = 32
tempo_dim = 8
temporal_only_cluster = True
EPS = 1e-15
recluster = False
save_path = TMP_PATH.joinpath('pyg-pinsage-tempo')

dataset = ml.StarWarsHomogenous()
data = dataset[0]
data = Data(data.node_stores[0].x, data.edge_stores[0].edge_index)
data.edge_stores[0].timestamp = dataset[0].edge_stores[0].timestamp
data = ToUndirected()(data)
data = SortEdges()(data)
snapshots = dataset.snapshots
G = dataset.G
G.to_undirected()

# Node Importance Definition
degree_avg = np.mean(dataset.G.degree())
row_ptrs, col_indices, perm = thg.data.to_csr(data.edge_index, data.num_nodes)
neighbor_weight = torch.ones(len(data.edge_index[0, :]))

walks_per_node = int(np.ceil(degree_avg))
walk_length = 5
start = torch.arange(data.num_nodes, dtype=torch.long)
walks = thg.algo.random_walk(row_ptrs, col_indices, start.repeat(walks_per_node), walk_length, p=1.0, q=1.0)

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

# neighbor_weight = torch.ones(len(data.edge_index[0, :]))
# data.edge_stores[0].weight = neighbor_weight

# Edges are always within snapshots (so take them all)
edge_index = data.edge_index
edge_snaps = data.timestamp

snapshot_nodes = []
for i, snapshot in enumerate(snapshots):
    snapshot_nodes.append(list(set(snapshot.node_stores[0].node_mask.nonzero().squeeze().tolist())))

snapshot_idx = set(range(len(snapshots)))
node_to_snapshot_idx = [
    {j for j, snap in enumerate(snapshots) if snap.node_stores[0].node_mask[i].item()}
    for i in range(data.num_nodes)
]


def pos_sample(nodes: torch.Tensor, num_samples: int = 5) -> torch.Tensor:
    pos_nodes = torch.tensor([
        [
            [nid, random.choice(list(node_to_snapshot_idx[i].intersection(node_to_snapshot_idx[nid])))]
            for nid in random.choices(list(set(it.chain(*[
            candidates for ct, candidates in enumerate(snapshot_nodes) if ct in node_to_snapshot_idx[i]
        ]))), k=num_samples)
        ]
        for i in nodes.tolist()
    ], dtype=torch.long)

    ctr_nodes = nodes.unsqueeze(1).repeat(1, num_samples).view(-1)
    pos_nodes = pos_nodes.view(-1, 2)
    pos_idx = torch.stack([ctr_nodes, pos_nodes[:, 0]], dim=1)
    pos_snaps = torch.stack([pos_nodes[:, 1], pos_nodes[:, 1]], dim=1)

    return pos_idx, pos_snaps


def neg_sample(nodes: torch.Tensor, snaps: torch.Tensor, num_neg_samples: int = 1) -> torch.Tensor:
    neg_nodes = torch.tensor([
        [
            [nid, random.choice(list(node_to_snapshot_idx[nid].difference([t])))]
            for nid in random.choices(list(set(it.chain(*[
            candidates for ct, candidates in enumerate(snapshot_nodes) if t != ct
        ]))), k=num_neg_samples)
        ]
        for i, t in zip(nodes.tolist(), snaps.tolist())
    ], dtype=torch.long)

    neg_idx = neg_nodes[:, :, 0]
    neg_snaps = neg_nodes[:, :, 1]

    return neg_idx, neg_snaps


repeat_count = 20
num_neg_samples = 3
# pos_idx = edge_index.t().repeat(repeat_count, 1)
# pos_snaps = torch.stack([
#     edge_snaps.repeat(repeat_count),
#     edge_snaps.repeat(repeat_count)
# ], dim=1)

pos_idx, pos_snaps = pos_sample(torch.arange(data.num_nodes), num_samples=repeat_count)
neg_idx, neg_snaps = neg_sample(pos_idx[:, 0], pos_snaps[:, 0], num_neg_samples=num_neg_samples)
data_node_idx = torch.cat([pos_idx, neg_idx], dim=1)
data_snaps = torch.cat([pos_snaps, neg_snaps], dim=1)
data_all = torch.stack([data_node_idx, data_snaps], dim=2)

window = (0, 0)


# Build dataloader
class TemporalWeightedNeighborLoader:
    def __init__(self, data: Data, num_neighbors: List[int]):
        self.num_neighbors = num_neighbors
        self.col_ptrs, self.row_indices, self.perm = thg.data.to_csc(data.edge_index, data.num_nodes)
        self.weights = data.edge_stores[0].weight[self.perm].double()
        self.timestamps = data.timestamp[self.perm].long()
        # super(NeighborLoader, self).__init__(data, batch_size, shuffle)

    def __call__(self, indices: List[int], snaps: List[int]):
        index = torch.tensor(indices)
        snaps = torch.tensor(snaps)
        nodes, rows, cols, edge_index, layer_offsets = thg.algo.neighbor_sampling_homogenous(
            self.col_ptrs, self.row_indices, index, self.num_neighbors,
            sampler=thg.WeightedSampler(self.weights),
            filter=thg.TemporalFilter(window, self.timestamps, snaps, forward=False, mode=thg.TEMPORAL_SAMPLE_RELATIVE)
        )
        return nodes, rows, cols, edge_index, len(indices), layer_offsets


class NodesLoader(BaseDataLoader):
    def __init__(self, data: Data, inputs: torch.Tensor, num_neighbors: List[int], **kwargs):
        self.data = data
        self.neighbor_sampler = TemporalWeightedNeighborLoader(data, num_neighbors)

        super().__init__(
            inputs.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long)
        node_idx = indices[:, :, 0].view(-1)
        node_snaps = indices[:, :, 1].view(-1)
        return self.neighbor_sampler(node_idx.tolist(), node_snaps.tolist())

    def transform_fn(self, out):
        nodes, rows, cols, edge_index, batch_size, layer_offsets = out
        data = filter_data(self.data, nodes, rows, cols, edge_index, self.neighbor_sampler.perm)
        data.batch_size = batch_size
        return data, layer_offsets


class SamplesLoader(NodesLoader):
    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long)
        node_idx = indices[:, :, 0].view(-1)
        node_snaps = indices[:, :, 1].view(-1)
        return self.neighbor_sampler(node_idx.tolist(), node_snaps.tolist())


num_neighbors = [4, 4]
batch_size = 8
nodes_loader = NodesLoader(
    data, torch.arange(data.num_nodes), num_neighbors,
    batch_size=batch_size, num_workers=0
)
data_loader = SamplesLoader(
    data, data_all, num_neighbors,
    batch_size=batch_size, num_workers=0, shuffle=True
)
batch, batch_layer_offsets = next(iter(data_loader))

node_loader = NeighborLoader(
    data=data, num_neighbors=[4, 4], directed=False, replace=False
)

# Build model
use_cosine = False
use_centers = True
initialize = None  # 'louvain'
# initialize = 'louvain'
n_clusters = 5
save = True

# n_epochs = 30  # 5 # 30
# comm_epoch = 10  # 2 #10
n_epochs = 50  # 5 # 30
comm_epoch = 30  # 2 #10


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

        self.conv1 = PinSAGELayer((32, 32), repr_dim, normalize)
        self.conv2 = PinSAGELayer((repr_dim, repr_dim), repr_dim, normalize)
        self.activation = torch.nn.ReLU()

    def forward(self, batch: Data, layer_offsets: List[Tuple[int, int]] = None) -> torch.Tensor:
        x = batch.x
        x = self.conv1(x, batch.edge_index, batch.weight)
        x = self.activation(x)
        x = self.conv2(x, batch.edge_index, batch.weight)
        return x[:batch.batch_size]


embedding_module = GraphSAGEHeteroModuleNew(
    repr_dim, n_layers=2, normalize=False
)
temporal_module = GraphSAGEHeteroModuleNew(
    tempo_dim, n_layers=2, normalize=False
)

ne_fn = NegativeEntropyRegularizer()

class MainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.temporal_module = temporal_module
        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)
        self.centroids_temp = torch.nn.Embedding(n_clusters, tempo_dim)

    def forward(self, batch, optimize_comms=False):
        batch, layer_offsets = batch
        emb_posi = self.embedding_module(batch, layer_offsets)
        emb_temp = self.temporal_module(batch, layer_offsets)

        emb_posi = emb_posi.view(-1, num_neg_samples + 2, repr_dim)
        emb_temp = emb_temp.view(-1, num_neg_samples + 2, tempo_dim)
        neg_pos_emb = torch.cat([emb_posi, emb_temp], dim=2)

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
            centroids = torch.cat([self.centroids.weight.clone(), self.centroids_temp.weight.clone()], dim=1)
            c_emb = centroids.unsqueeze(0).transpose(1, 2).repeat(current_batch_size, 1, 1)
            c_ctr_q = torch.softmax(torch.bmm(ctr_emb, c_emb), dim=-1)
            c_pos_q = torch.softmax(torch.bmm(pos_emb, c_emb), dim=-1)
            c_neg_q = torch.softmax(torch.bmm(neg_emb, c_emb), dim=-1)

            c_pos_d = torch.bmm(c_ctr_q, c_pos_q.transpose(1, 2)).view(-1)
            c_neg_d = torch.bmm(c_ctr_q, c_neg_q.transpose(1, 2)).max(dim=2).values.view(-1)
            c_mm_loss = torch.clip(c_neg_d - c_pos_d + 1, min=0).mean()
            loss = mm_loss + c_mm_loss

            q = torch.cat([c_ctr_q, c_pos_q, c_neg_q], dim=1).view(-1, n_clusters)
            ne = ne_fn(q)
            loss = mm_loss + c_mm_loss + ne * 0.01


            # KL Divergence loss (Confidence improvement)
            # q = torch.cat([c_ctr_q, c_pos_q, c_neg_q], dim=1).view(-1, n_clusters)
            # p = q.detach().square()
            # ca_loss = kl_loss(torch.log(q), p)
            # ne = ne_fn(q)
            # loss = c_mm_loss + mm_loss + ca_loss * 0.01 + ne * 0.01
            u = 0

        return loss



model = MainModel()
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

embedding_module.load_state_dict(torch.load(str(TMP_PATH.joinpath('pyg-pinsage/embeddings.pt'))))
embedding_module.requires_grad_(False)

model.centroids.load_state_dict(torch.load(str(TMP_PATH.joinpath('pyg-pinsage/centroids.pt'))))
model.centroids.requires_grad_(False)


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
    for batch in node_loader:
        emb_posi = embedding_module(batch)
        emb_temp = temporal_module(batch)
        embs.append(torch.cat([emb_posi, emb_temp], dim=1))

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

if not recluster:
    print('Reusing trained centers')
    centers = torch.cat([
        model.centroids.weight.clone(), model.centroids_temp.weight.clone()
    ], dim=1).detach()
    q = torch.softmax(torch.mm(embeddings, centers.transpose(1, 0)), dim=-1)
    I = q.argmax(dim=-1)
    I = I.numpy()
else:
    if temporal_only_cluster:
        embeddings = embeddings[:, repr_dim:]

    if use_cosine:
        print('Normalize for cosine similarity')
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    print('Searching for cluster centers using K-means')
    kmeans = faiss.Kmeans(embeddings.shape[1], k=n_clusters, niter=20, verbose=True, nredo=10)
    em = np.ascontiguousarray(embeddings.numpy())
    kmeans.train(em)
    D, I = kmeans.index.search(em, 1)


if save:
    save_projector("Star Wars Temporal", embeddings, pd.DataFrame({
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
G = DataGraph.from_schema(schema)
G.to_undirected()

metrics = get_metric_list(ground_truth=False, overlapping=False)
results = pd.DataFrame([
    {
        'metric': metric_cls.metric_name(),
        'value': metric_cls.calculate(G, comlist)
    }
    for metric_cls in metrics]
)
print(results)
