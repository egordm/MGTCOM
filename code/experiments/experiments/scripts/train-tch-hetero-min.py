import pytorch_lightning as pl
import torch

from ml import newman_girvan_modularity, igraph_from_hetero
from ml.datasets import DBLPHCN, StarWars, IMDB5000
from ml.layers import ExplicitClusteringModule
from ml.layers.embedding import HGTModule
from ml.layers.initialization import LouvainInitializer
from ml.models.positional import PositionalModel, PositionalDataModule
from shared.constants import BENCHMARKS_RESULTS

# dataset = StarWars()
# batch_size = 16
# n_clusters = 5
dataset = IMDB5000()
batch_size = 512
n_clusters = 50
# dataset = DBLPHCN()
# batch_size = 512
# n_clusters = 55
# n_clusters = 70
# n_clusters = 5

data = dataset.data

name = f'tch-hetero-min-{dataset.name}'
save_dir = BENCHMARKS_RESULTS.joinpath('analysis', name)
save_dir.mkdir(parents=True, exist_ok=True)

lr = 0.01
lr_cosine=False
repr_dim = 32
# repr_dim = 64
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

loader = PositionalDataModule(
    data, num_samples=num_samples, num_neg_samples=num_neg_samples, batch_size=batch_size, temporal=temporal,
    num_workers=workers, prefetch_factor=4 if workers else 2, persistent_workers=True if workers else False,
)

# embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)
embedding_module = HGTModule(data.metadata(), repr_dim, num_heads=2, num_layers=2, use_RTE=temporal)
clustering_module = ExplicitClusteringModule(repr_dim, n_clusters, sim=sim)
model = PositionalModel(embedding_module, clustering_module, lr=lr, lr_cosine=lr_cosine, c_weight=c_weight, ne_weight=ne_weight, sim=sim)

initializer = LouvainInitializer(data)

# Pretraining
print("Pretraining")
trainer = pl.Trainer(gpus=gpus, min_epochs=n_epochs, max_epochs=n_epochs, callbacks=callbacks)
trainer.fit(model, loader)

# Reinitialize
print("Initialize")
embeddings = model.compute_embeddings(trainer, loader)
centers = initializer.initialize(embeddings)
model.clustering_module.reinit(centers)

# Cluster-aware training
model.use_clustering = True
trainer = pl.Trainer(gpus=gpus, min_epochs=n_comm_epochs, max_epochs=n_comm_epochs, callbacks=callbacks, enable_model_summary=False)
trainer.fit(model, loader)

print('Calculating node assignments')
embeddings = model.compute_embeddings(trainer, loader)
I = {k: model.clustering_module.assign(emb).detach().cpu() for k, emb in embeddings.items()}

m = newman_girvan_modularity(data, I, clustering_module.n_clusters)
print(f'Modularity: {m:.4f}')

G, _, _ = igraph_from_hetero(data, node_attrs=dict(comm=I))
G.write_graphml(str(save_dir.joinpath('graph.graphml')))

torch.save(model.state_dict(), str(save_dir.joinpath('model.pt')))
