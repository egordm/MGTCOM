import pytorch_lightning as pl
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import Node2Vec

import ml
from experiments import cosine_cdist, euclidean_cdist
from shared.constants import TMP_PATH

device = 'cpu'
experiment_name = 'pyg-n2v-experiment'
node_type = 'Character'
initialization = 'louvain'  # 'k-means' or 'none
repr_dim = 32
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
G = dataset.G
G.to_undirected()



def initial_clustering(embeddings: torch.Tensor):
    clustering = G.community_multilevel()
    assignment = torch.tensor(clustering.membership)
    cluster_count = len(clustering)
    assigned_count = torch.zeros(cluster_count, dtype=torch.long).scatter_add(0, assignment, torch.ones_like(assignment,
                                                                                                             dtype=torch.long))
    centroids = torch.zeros(cluster_count, repr_dim, dtype=torch.float).index_add_(0, assignment, embeddings)
    centroids = centroids / assigned_count.unsqueeze(1)
    return centroids


class Model(Node2Vec):
    def __init__(self, edge_index, embedding_dim, walk_length, context_size, walks_per_node=1, p=1, q=1,
                 num_negative_samples=1, num_nodes=None, sparse=False, n_clusters=5):
        super().__init__(edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q,
                         num_negative_samples, num_nodes, sparse)
        self.n_clusters = n_clusters
        self.centroids = torch.nn.Embedding(n_clusters, embedding_dim)
        self.cos_sim = torch.nn.CosineSimilarity(dim=2)
        self.euc_dist = torch.nn.PairwiseDistance(p=2)
        self.is_pretraining = True

    def loss(self, pos_rw, neg_rw):
        hp_loss = self.homophily_loss(pos_rw, neg_rw)
        if not self.is_pretraining:
            cc_loss = self.clustering_loss(pos_rw, neg_rw)
            return {
                'loss': hp_loss + cc_loss,
                'hp_loss': hp_loss.detach(),
                'cc_loss': cc_loss.detach()
            }
        else:
            return {
                'loss': hp_loss,
                'hp_loss': hp_loss.detach()
            }

    def homophily_loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        # out = self.cos_sim(h_start, h_rest).view(-1)
        out = -self.euc_dist(h_start, h_rest).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        # out = self.cos_sim(h_start, h_rest).view(-1)
        out = -self.euc_dist(h_start, h_rest).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def clustering_loss(self, pos_rw, neg_rw):
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        q_start = cosine_cdist(h_start, self.centroids.weight.clone().unsqueeze(0), dim=2)
        # q_start = -euclidean_cdist(h_start, self.centroids.weight.clone().unsqueeze(0))
        q_start = torch.softmax(q_start, dim=2)

        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)
        q_rest = cosine_cdist(h_rest, self.centroids.weight.clone().unsqueeze(0), dim=2)
        # q_rest = -euclidean_cdist(h_rest, self.centroids.weight.clone().unsqueeze(0))
        q_rest = torch.mean(q_rest, dim=1, keepdim=True)
        q_rest = torch.softmax(q_rest, dim=2)

        out = (q_start * q_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        q_start = cosine_cdist(h_start, self.centroids.weight.clone().unsqueeze(0), dim=2)
        # q_start = -euclidean_cdist(h_start, self.centroids.weight.clone().unsqueeze(0))
        q_start = torch.softmax(q_start, dim=2)

        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)
        q_rest = cosine_cdist(h_rest, self.centroids.weight.clone().unsqueeze(0), dim=2)
        # q_rest = -euclidean_cdist(h_rest, self.centroids.weight.clone().unsqueeze(0))
        q_rest = torch.mean(q_rest, dim=1, keepdim=True)
        q_rest = torch.softmax(q_rest, dim=2)

        out = (q_start * q_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Node2Vec(data[data.edge_types[0]].edge_index, embedding_dim=repr_dim, walk_length=8,
#                  context_size=8, walks_per_node=10,
#                  num_negative_samples=3, p=1, q=1, sparse=True).to(device)
model = Model(data[data.edge_types[0]].edge_index, embedding_dim=repr_dim, walk_length=8,
              context_size=8, walks_per_node=10,
              num_negative_samples=3, p=1, q=1, sparse=False, n_clusters=5).to(device)

loader = model.loader(batch_size=16, shuffle=True, num_workers=0)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

import numpy as np


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        outs = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss = outs['loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z, data.y,
                     z, data.y,
                     max_iter=150)
    return acc


for epoch in range(1, 500):
    if epoch >= 200:
        model.is_pretraining = False
    if epoch == 200:
        embeddings = model.embedding.weight.detach().cpu()
        centroids = initial_clustering(embeddings)
        model.centroids = torch.nn.Embedding.from_pretrained(centroids, freeze=False)

    loss = train()
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #

from shared.constants import BENCHMARKS_RESULTS
import faiss
import pandas as pd

save_path = BENCHMARKS_RESULTS.joinpath('analysis', experiment_name)
save_path.mkdir(parents=True, exist_ok=True)

embeddings = model.embedding.weight.detach().cpu()
centroids = model.centroids.weight.detach().cpu()

if recluster:
    k = 5
    kmeans = faiss.Kmeans(embeddings.shape[1], k, niter=20, verbose=True, nredo=10)
    kmeans.train(embeddings.numpy())
    D, I = kmeans.index.search(embeddings.numpy(), 1)
else:
    sim = cosine_cdist(embeddings, centroids)
    I = torch.argmax(sim, dim=1).numpy()

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
