from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import random
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.base import BaseDataLoader
from scipy.sparse import csr_matrix
import itertools as it
from torch.utils.tensorboard import SummaryWriter

import experiments
import ml
from experiments import cosine_cdist, euclidean_cdist, NegativeEntropyRegularizer, save_projector
from shared.constants import TMP_PATH

device = 'cpu'
experiment_name = 'pyg-sage-sim-tempo'
node_type = 'Character'
# initialization = 'louvain'  # 'k-means' or 'none
repr_dim = 32
tempo_dim = 8
EPS = 1e-15
recluster = False
save_path = TMP_PATH.joinpath(experiment_name)
callbacks = [
    pl.callbacks.ModelSummary(),
    pl.callbacks.LearningRateMonitor(),
    pl.callbacks.EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")
]

dataset = ml.StarWarsHomogenous()
transform = ToUndirected()
data = dataset[0]
snapshots = dataset.snapshots
G = dataset.G
G.to_undirected()

# Edges are always within snapshots (so take them all)
edge_index = data.edge_stores[0].edge_index
edge_index = torch.cat([
    edge_index,
    torch.stack([edge_index[1, :], edge_index[0, :]])
], dim=1)
edge_snaps = torch.cat([
    data.edge_stores[0].timestamp,
    data.edge_stores[0].timestamp
], dim=0) - 1

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

snapshot_loaders = [
    NeighborLoader(
        data=snapshot, num_neighbors=[4, 4], input_nodes='Character', directed=False, replace=False
    )
    for i, snapshot in enumerate(snapshots)
]


class TemporalSamplesLoader(BaseDataLoader):
    def __init__(
            self,
            data,
            **kwargs
    ):
        self.snapshot_loaders = snapshot_loaders

        super().__init__(
            data.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long)
        node_idx = indices[:, :, 0].view(-1)
        node_snaps = indices[:, :, 1].view(-1)
        node_batch_idx = torch.arange(node_idx.size(0), dtype=torch.long)
        grouped_node_idx = defaultdict(list)
        grouped_batch_idx = defaultdict(list)
        for i, j, k in zip(node_snaps.tolist(), node_idx.tolist(), node_batch_idx.tolist()):
            grouped_node_idx[i].append(j)
            grouped_batch_idx[i].append(k)

        return {
            s: {
                'node_idx': self.snapshot_loaders[s].neighbor_sampler(grouped_node_idx[s]),
                'batch_idx': torch.tensor(grouped_batch_idx[s], dtype=torch.long)
            }
            for s in grouped_node_idx.keys()
        }

    def transform_fn(self, out):
        return {
            s: {
                'node_data': self.snapshot_loaders[s].transform_fn(v['node_idx']),
                'batch_idx': v['batch_idx'],
            }
            for s, v in out.items()
        }


batch_size = 8
data_loader = TemporalSamplesLoader(data_all, batch_size=batch_size, shuffle=True)
batch = next(iter(data_loader))

node_loader = NeighborLoader(
    data=data, num_neighbors=[4, 4], input_nodes='Character', directed=False, replace=False
)

embedding_module = experiments.GraphSAGEModule(node_type, data.metadata(), repr_dim, n_layers=2)
temporal_module = experiments.GraphSAGEModule(node_type, data.metadata(), tempo_dim, n_layers=2)

use_cosine = False
temporal_only_cluster = True
initialize = 'louvain'
n_clusters = 8


class MainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.temporal_module = temporal_module
        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)
        self.centroids_temp = torch.nn.Embedding(n_clusters, tempo_dim)

    def forward(self, batch, optimize_comms=False):
        emb_posi = torch.zeros(sum(len(v['batch_idx']) for v in batch.values()), repr_dim)
        emb_temp = torch.zeros(sum(len(v['batch_idx']) for v in batch.values()), tempo_dim)
        for k, v in batch.items():
            v_posi = self.embedding_module(v['node_data'])
            v_temp = self.temporal_module(v['node_data'])
            emb_posi = emb_posi.index_add(0, v['batch_idx'], v_posi)
            emb_temp = emb_temp.index_add(0, v['batch_idx'], v_temp)

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

embedding_module.load_state_dict(torch.load(str(TMP_PATH.joinpath('pyg-sage-sim/embeddings.pt'))))
embedding_module.freeze()
embedding_module.requires_grad_(False)

model.centroids.load_state_dict(torch.load(str(TMP_PATH.joinpath('pyg-sage-sim/centroids.pt'))))
model.centroids.requires_grad_(False)

n_epochs = 30  # 5 # 30
comm_epoch = 10  # 2 #10
# n_epochs = 20  # 5 # 30
# comm_epoch = 10  # 2 #10
# n_epochs = 40  # 5 # 30
# comm_epoch = 20  # 2 #10
# n_epochs = 5  # 30
# comm_epoch = 2  # 10


def train(epoch):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        loss = model(batch, optimize_comms=(epoch >= comm_epoch))

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
    loss = train(epoch)
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #

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
