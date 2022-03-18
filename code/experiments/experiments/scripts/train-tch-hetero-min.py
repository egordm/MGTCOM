import pytorch_lightning as pl
import torch

from ml import SortEdges, newman_girvan_modularity, igraph_from_hetero
from ml.datasets import IMDB5000, StarWars, DBLPHCN
from ml.layers import ExplicitClusteringModule
from ml.layers.embedding import HGTModule
from ml.layers.initialization import LouvainInitializer, KMeansInitializer
from ml.models.positional import PositionalModel, PositionalDataModule
from ml.transforms.undirected import ToUndirected
from ml.utils.collections import merge_dicts
from shared.constants import BENCHMARKS_RESULTS

# dataset = StarWars()
# batch_size = 16
# n_clusters = 5
# dataset = IMDB5000()
# batch_size = 512
# n_clusters = 50
dataset = DBLPHCN()
batch_size = 512
n_clusters = 55
# n_clusters = 70
# n_clusters = 5

data = dataset.data

name = f'tch-hetero-min-{dataset.name}'
lr = 0.01
lr_cosine=False
# repr_dim = 32
repr_dim = 64
# n_epochs = 20
# n_comm_epochs = 10
n_epochs = 8
n_comm_epochs = 6
# n_epochs = 1
# n_comm_epochs = 1
num_samples = [4, 3]
num_neg_samples = 3
ne_weight = 0.001
c_weight = 1.0
sim = 'dotp'
# sim = 'cosine'

temporal = False
gpus = 1
workers = 8
# gpus = 0
# workers = 0

callbacks = [
    pl.callbacks.LearningRateMonitor(logging_interval='step'),
]

data_module = PositionalDataModule(
    data, num_samples=num_samples, num_neg_samples=num_neg_samples, batch_size=batch_size, temporal=temporal,
    num_workers=workers, prefetch_factor=4 if workers else 2, persistent_workers=True if workers else False,
)

# embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)
embedding_module = HGTModule(data.metadata(), repr_dim, num_heads=2, num_layers=2, use_RTE=temporal)
clustering_module = ExplicitClusteringModule(repr_dim, n_clusters, sim=sim)
model = PositionalModel(embedding_module, clustering_module, lr=lr, lr_cosine=lr_cosine, c_weight=c_weight, ne_weight=ne_weight, sim=sim)

# Pretraining
trainer = pl.Trainer(gpus=gpus, min_epochs=n_epochs, max_epochs=n_epochs, callbacks=callbacks)
trainer.fit(model, data_module)

# Get Embeddings
pred = trainer.predict(model, data_module)
embeddings = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

# Reinitialize
initializer = LouvainInitializer(data)
# initializer = KMeansInitialization(data, n_clusters)
centers = initializer.initialize(embeddings)
clustering_module.reinit(centers)

# Cluster-aware training
model.use_clustering = True
# embedding_module.requires_grad_(False)
trainer = pl.Trainer(gpus=gpus, min_epochs=n_comm_epochs, max_epochs=n_comm_epochs, callbacks=callbacks, enable_model_summary=False)
trainer.fit(model, data_module)

# Get Embeddings
pred = trainer.predict(model, data_module)
embeddings = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

print('Reusing trained centers')
I = {k: clustering_module.assign(emb).detach().cpu() for k, emb in embeddings.items()}

m = newman_girvan_modularity(data, I, clustering_module.n_clusters)
print(f'Modularity: {m:.4f}')

save_dir = BENCHMARKS_RESULTS.joinpath('analysis', name)
save_dir.mkdir(parents=True, exist_ok=True)
G, _, _ = igraph_from_hetero(data, node_attrs=dict(comm=I))
G.write_graphml(str(save_dir.joinpath('graph.graphml')))