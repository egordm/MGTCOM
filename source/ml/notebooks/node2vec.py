import torch
from pytorch_lightning import Trainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from datasets import StarWars
from ml.callbacks.progress_bar import CustomProgressBar
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.node2vec_sampler import Node2VecSampler
from ml.layers.embedding.hetero_embedding import NodeEmbedding
from ml.models.node2vec import Node2VecModel
from ml.utils import Metric

dataset = StarWars()
data = dataset.data
hdata = data.to_homogeneous()

sampler = Node2VecSampler(hdata.edge_index, walk_length=8, walks_per_node=8, context_size=4, num_nodes=hdata.num_nodes)

loader = NodesLoader(hdata.num_nodes, transform=sampler, batch_size=16, num_workers=0)
test = next(iter(loader))

model = Node2VecModel(
    embedder=NodeEmbedding(hdata.num_nodes, 2),
    metric=Metric.L2,
    hparams={'lr': 0.01}
)

bar = CustomProgressBar()

trainer = Trainer(gpus=None, max_epochs=30, callbacks=[bar])
trainer.fit(model, loader)

z = model(torch.arange(hdata.num_nodes))
z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())

plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], s=20)
plt.axis('off')
plt.show()
