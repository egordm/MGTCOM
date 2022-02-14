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

import experiments
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

edge_index = data.edge_stores[0].edge_index
edge_index = torch.cat([
    edge_index,
    torch.stack([edge_index[1, :], edge_index[0, :]])
], dim=1)
row, col = edge_index

csr = csr_matrix(
    (torch.ones(edge_index.shape[1], dtype=torch.int32).numpy(), (row.numpy(), col.numpy())),
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


repeat_count = 2
num_neg_samples = 3
pos_idx = edge_index.t().repeat(repeat_count, 1)
neg_idx = neg_sample(pos_idx, num_neg_samples=num_neg_samples)
data_idx = torch.cat([pos_idx, neg_idx], dim=1)

node_loader = NeighborLoader(
    data=data, num_neighbors=[4, 4], input_nodes='Character', directed=False, replace=False
)


# data_idx[:2, :].view(-1).tolist()
# node_loader.transform_fn(node_loader.neighbor_sampler(data_idx[:2, :].view(-1).tolist()))

class SamplesLoader(BaseDataLoader):
    def __init__(
            self,
            data,
            **kwargs
    ):
        self.node_loader = node_loader

        super().__init__(
            data.tolist(),
            collate_fn=self.sample,
            **kwargs,
        )

    def sample(self, indices):
        indices = torch.tensor(indices, dtype=torch.long).view(-1).tolist()
        return self.node_loader.neighbor_sampler(indices)

    def transform_fn(self, out):
        return self.node_loader.transform_fn(out)


data_loader = SamplesLoader(data_idx, batch_size=8, shuffle=True)
batch = next(iter(data_loader))

embedding_module = experiments.GraphSAGEModule(node_type, data.metadata(), repr_dim, n_layers=2)
model = embedding_module
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

cos_sim = torch.nn.CosineSimilarity(dim=2)


def train():
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        emb = embedding_module(batch)
        neg_pos_emb = emb.view(-1, num_neg_samples + 2, repr_dim)
        ctr_emb = neg_pos_emb[:, 0, :].unsqueeze(1)
        pos_emb = neg_pos_emb[:, 1, :].unsqueeze(1)
        neg_emb = neg_pos_emb[:, 2:, :]

        out = cos_sim(ctr_emb, pos_emb).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        out = cos_sim(ctr_emb, neg_emb).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def get_embeddings():
    model.eval()
    embs = []
    for batch in node_loader:
        emb = embedding_module(batch)
        embs.append(emb)

    return torch.cat(embs, dim=0)


for epoch in range(1, 20):
    loss = train()
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #

u = 0
# aaaa

from shared.constants import BENCHMARKS_RESULTS
import faiss
import pandas as pd

save_path = BENCHMARKS_RESULTS.joinpath('analysis', experiment_name)
save_path.mkdir(parents=True, exist_ok=True)

embeddings = get_embeddings().detach()

# Normalize for cosine similarity
embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

k = 5
kmeans = faiss.Kmeans(embeddings.shape[1], k, niter=20, verbose=True, nredo=10)
kmeans.train(embeddings.numpy())
D, I = kmeans.index.search(embeddings.numpy(), 1)

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
