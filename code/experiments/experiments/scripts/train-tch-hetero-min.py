import pytorch_lightning as pl
import torch
from tch_geometric.loader import CustomLoader
from tch_geometric.loader.hgt_loader import HGTLoader
from tch_geometric.transforms import NegativeSamplerTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform
from torch_geometric.transforms import ToUndirected

from experiments.datasets import StarWars
from ml import SortEdges, newman_girvan_modularity
from ml.layers import ExplicitClusteringModule
from ml.layers.embedding import HGTModule
from ml.loaders.contrastive_dataloader import ContrastiveDataLoader
from ml.loaders.dataset import HeteroEdgesDataset, HeteroNodesDataset
from ml.models.positional import PositionalModel
from ml.utils.collections import merge_dicts

dataset = StarWars()
data = dataset[0]
data = ToUndirected(reduce='max')(data)
data = SortEdges()(data)
G = dataset.G
G.to_undirected()

repr_dim = 32
n_epochs = 20  # 10
n_comm_epochs = 10
# n_epochs = 1  # 10
# n_comm_epochs = 1
n_clusters = 5
batch_size = 16
temporal = False


neg_sampler = NegativeSamplerTransform(data, 3, 5)
# neighbor_sampler = NeighborSamplerTransform(data, [4, 3])
# neighbor_sampler = NeighborSamplerTransform(data, [3, 2])
# neighbor_sampler = HGTSamplerTransformz(data, [3, 2])
# neighbor_sampler = HGTSamplerTransform(data, [3, 2])
neighbor_sampler = HGTSamplerTransform(data, [3, 2], temporal=temporal)

data_loader = ContrastiveDataLoader(
    HeteroEdgesDataset(data, temporal=temporal),
    neg_sampler, neighbor_sampler, batch_size=batch_size, shuffle=True
)
nodes_loader = HGTLoader(
    HeteroNodesDataset(data, temporal=temporal),
    neighbor_sampler=neighbor_sampler,
    batch_size=batch_size, shuffle=False
)

# embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)
embedding_module = HGTModule(data.metadata(), repr_dim, repr_dim, num_heads=2, num_layers=2, use_RTE=temporal)
clustering_module = ExplicitClusteringModule(repr_dim, n_clusters)
model = PositionalModel(embedding_module, clustering_module)

# Pretraining
trainer = pl.Trainer(gpus=0, min_epochs=n_epochs, max_epochs=n_epochs)
trainer.fit(model, data_loader)
# Cluster-aware training
model.use_clustering = True
trainer = pl.Trainer(gpus=0, min_epochs=n_comm_epochs, max_epochs=n_comm_epochs, enable_model_summary=False)
trainer.fit(model, data_loader)

# Get Embeddings
pred = trainer.predict(model, nodes_loader)
embeddings = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

print('Reusing trained centers')
emb_q = {k: clustering_module(emb) for k, emb in embeddings.items()}
I = {k: q.argmax(dim=-1) for k, q in emb_q.items()}

m = newman_girvan_modularity(data, I, n_clusters)
print(f'Modularity: {m:.4f}')