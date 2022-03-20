import pytorch_lightning as pl
import torch

from ml import newman_girvan_modularity, igraph_from_hetero
from ml.datasets import DBLPHCN, StarWars
from ml.layers import ExplicitClusteringModule
from ml.layers.embedding import HGTModule
from ml.layers.initialization import LouvainInitializer
from ml.models import TemporalDataModule, TemporalModel
from ml.models.positional import PositionalModel, PositionalDataModule
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

name = f'tch-hetero-min-tempo-{dataset.name}'
save_dir = BENCHMARKS_RESULTS.joinpath('analysis', name)
save_dir.mkdir(parents=True, exist_ok=True)
name = f'tch-hetero-min-{dataset.name}'
load_dir = BENCHMARKS_RESULTS.joinpath('analysis', name)
if not load_dir.exists():
    load_dir = None
    raise FileNotFoundError(f'{load_dir} does not exist')

lr = 0.01
lr_cosine = False
repr_dim = 32
# repr_dim = 64
# n_epochs = 20
# n_comm_epochs = 10
n_epochs = 8
# n_comm_epochs = 6
# n_epochs = 1
# n_comm_epochs = 1
num_samples = [4, 3]
num_neg_samples = 3
ne_weight = 0.001
c_weight = 1.0
sim = 'dotp'
# sim = 'cosine'

temporal = False
# window = (0, 1)
window_fraction = 0.2
repeat_count = 8
temp_dim = 8

gpus = 1
workers = 8
# gpus = 0
# workers = 0

# Load state
if load_dir is not None:
    load_state = torch.load(load_dir.joinpath('model.pt'))
    loaded_centroids = load_state['clustering_module.centroids.weight']
    n_clusters, repr_dim = loaded_centroids.shape
    print(f'Loaded n_clusters={n_clusters} clusters of repr_dim={repr_dim} dimensions')

# Determine window size
timestamps = torch.cat(list(data.timestamp_dict.values()), dim=0).float().sort()\
    .values.quantile(torch.tensor([0.05, 0.95]))
interval = int(torch.ceil((timestamps[1] - timestamps[0]) * window_fraction).detach().item())
window = (0, interval) # window is relative to timestamp, but is integer
print(f'Relative Window size: window={window}')

callbacks = [
    pl.callbacks.LearningRateMonitor(logging_interval='step'),
]

loader = TemporalDataModule(
    data, num_samples=num_samples, window=window, repeat_count=repeat_count, batch_size=batch_size,
    num_workers=workers, prefetch_factor=4 if workers else 2, persistent_workers=True if workers else False,
)

# embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)
embedding_module = HGTModule(data.metadata(), repr_dim, num_heads=2, num_layers=2, use_RTE=temporal)
temp_embedding_module = HGTModule(data.metadata(), temp_dim, num_heads=1, num_layers=2, use_RTE=temporal)
clustering_module = ExplicitClusteringModule(repr_dim, n_clusters, sim=sim)
temp_clustering_module = ExplicitClusteringModule(temp_dim, n_clusters, sim=sim)

model = TemporalModel(
    embedding_module, temp_embedding_module, clustering_module, temp_clustering_module,
    lr=lr, lr_cosine=lr_cosine, c_weight=c_weight, ne_weight=ne_weight, sim=sim
)

if load_dir:
    state = model.state_dict()
    state.update(load_state)
    load_res = model.load_state_dict(state)

embedding_module.requires_grad_(False)
clustering_module.requires_grad_(False)

# Cluster-aware training
model.use_clustering = True
trainer = pl.Trainer(gpus=gpus, min_epochs=n_epochs, max_epochs=n_epochs, callbacks=callbacks,
                     enable_model_summary=True)
trainer.fit(model, loader)

print('Calculating node assignments')
embeddings = model.compute_embeddings(trainer, loader)

I = {k: model.compute_assignments(emb).detach().cpu() for k, emb in embeddings.items()}

m = newman_girvan_modularity(data, I, clustering_module.n_clusters)
print(f'Modularity: {m:.4f}')

G, _, _ = igraph_from_hetero(data, node_attrs=dict(comm=I))
G.write_graphml(str(save_dir.joinpath('graph.graphml')))
