import pytorch_lightning as pl
import torch

from ml import igraph_from_hetero
from experiments.algo import louvain, loucom
from ml.datasets import StarWars
from ml.layers import ExplicitClusteringModule, pairwise_sim_fn
from ml.layers.embedding import HGTModule
from ml.models.positional import PositionalModel, PositionalDataModule
from ml.transforms.typed_homogenous import TypedHomogenousTransform
from shared.constants import BENCHMARKS_RESULTS

dataset = StarWars()
batch_size = 16
n_clusters = 5
# dataset = IMDB5000()
# batch_size = 512
# n_clusters = 50
# dataset = DBLPHCN()
# batch_size = 512
# n_clusters = 55
# n_clusters = 70
# n_clusters = 5

data = dataset.data

G, _, _, node_offsets = igraph_from_hetero(data)
comm = G.community_multilevel()
print(f'Found communities with modularity={comm.modularity}')

uni_data = TypedHomogenousTransform()(data)
centroids = torch.rand(uni_data.num_nodes, 2)
ms, qs = louvain(
    uni_data.num_nodes, uni_data.edge_index, torch.ones(uni_data.num_edges),
)
print(f'Own modularity={qs[-1]}, Real = {G.modularity(ms[-1])}')

ms, qs, cc, cs = loucom(
    uni_data.num_nodes, centroids, uni_data.edge_index, torch.ones(uni_data.num_edges),
)
print(f'Own modularity={qs[-1]}, Real = {G.modularity(ms[-1])}')
# ccs = torch.zeros(cc[-1]).index_add(0, ms[-1], torch.ones(ms[-1].shape))
# css = torch.zeros(cc[-1], 2).index_add(0, ms[-1], centroids) / ccs.unsqueeze(1)

# u = np.array(G.get_edgelist())[G.incident(0)]
# exit(0)


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

# gpus = 1
# workers = 8
gpus = 0
workers = 0

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
model = PositionalModel(embedding_module, clustering_module, lr=lr, lr_cosine=lr_cosine, c_weight=c_weight,
                        ne_weight=ne_weight, sim=sim)

# Load state
if load_dir is not None:
    load_state = torch.load(load_dir.joinpath('model.pt'))
    loaded_centroids = load_state['clustering_module.centroids.weight']
    n_clusters, repr_dim = loaded_centroids.shape
    print(f'Loaded n_clusters={n_clusters} clusters of repr_dim={repr_dim} dimensions')

print('Calculating node embeddings')
trainer = pl.Trainer(gpus=gpus, min_epochs=n_epochs, max_epochs=n_epochs, callbacks=callbacks)
emb_dict = model.compute_embeddings(trainer, loader)

G, _, _, node_offsets = igraph_from_hetero(data)

embeddings = torch.zeros((G.vcount(), repr_dim))
for node_type, offset in node_offsets.items():
    embeddings[offset:offset + len(emb_dict[node_type])] = emb_dict[node_type]

edges = torch.tensor(G.get_edgelist(), dtype=torch.long)
sim_fn = pairwise_sim_fn(sim)
weights = sim_fn(embeddings[edges[:, 0]], embeddings[edges[:, 1]]).numpy()

comm = G.community_multilevel(weights)
G.vs['comm_id'] = comm.membership
G.es['weight'] = weights

print(f'Weighted modularity={comm.modularity}')
m = G.modularity(comm.membership)
print(f'Unweighted modularity={m}')

# G.write_graphml(str(load_dir.joinpath('graph_weighted.graphml')))

ms, qs, cc, cs = loucom(
    G.vcount(), embeddings, edges.t(), torch.ones(len(edges)),
)
print(f'Own modularity={qs[-1]}, Real = {G.modularity(ms[-1])}')