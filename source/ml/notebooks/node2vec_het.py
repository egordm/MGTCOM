from typing import Dict

import torch
from pytorch_lightning import Trainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from datasets import StarWars
from ml.callbacks.progress_bar import CustomProgressBar
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.hgt_sampler import HGTSampler, HGTSamplerParams
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.eval_split import EvalNodeSplitTransform
from ml.layers.conv.hybrid_conv_net import HybridConvNet, HybridConvNetParams
from ml.models.node2vec import Node2VecModel
from ml.utils import Metric

dataset = StarWars()
train_data, val_data, test_data = EvalNodeSplitTransform()(dataset.data)
train_hdata = train_data.to_homogeneous()


n2v_sampler = Node2VecSampler(
    train_hdata.edge_index, train_hdata.num_nodes,
    hparams=Node2VecSamplerParams(walk_length=8, walks_per_node=8, context_size=4)
)
hgt_sampler = HGTSampler(train_data, hparams=HGTSamplerParams(num_samples=[2, 3]))
sampler = HybridSampler(n2v_sampler=n2v_sampler, hgt_sampler=hgt_sampler)

loader = NodesLoader(train_hdata.num_nodes, transform=sampler, batch_size=16, num_workers=0)
test = next(iter(loader))


class HetEmbed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = HybridConvNet(
            train_data.metadata(),
            embed_num_nodes=train_data.num_nodes_dict,
            # embed_num_nodes={},
            hparams=HybridConvNetParams(repr_dim=2, num_layers=2)
        )

    def forward(self, batch: HeteroData) -> Tensor:
        Z_dict = self.embedder(batch)

        Z = torch.zeros(batch.batch_size, self.embedder.repr_dim)
        for store in batch.node_stores:
            node_type = store._key
            Z[store.batch_perm] = Z_dict[node_type]

        return Z


model = Node2VecModel(
    embedder=HetEmbed(),
    metric=Metric.L2,
    hparams={'lr': 0.01}
)

bar = CustomProgressBar()

trainer = Trainer(gpus=None, max_epochs=5, callbacks=[bar])
trainer.fit(model, loader)

test_hgt_sampler = HGTSampler(test_data, hparams=HGTSamplerParams(num_samples=[2, 3]))
pred_data = test_hgt_sampler({
    'Character': torch.arange(test_data['Character'].num_nodes)
})
z = model(pred_data)
z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())

plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], s=20)
plt.axis('off')
plt.show()
