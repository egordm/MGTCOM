import torch
from pytorch_lightning import Trainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import Tensor
from torch_geometric.data import HeteroData

from datasets import StarWars
from datasets.utils.conversion import igraph_from_hetero
from ml.callbacks.progress_bar import CustomProgressBar
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.ballroom_sampler import BallroomSampler, BallroomSamplerParams, TemporalNodeIndex
from ml.data.samplers.hgt_sampler import HGTSampler, HGTSamplerParams
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.transforms.eval_split import EvalNodeSplitTransform
from ml.layers.fc_net import FCNet, FCNetParams
from ml.layers.conv.hybrid_conv_net import HybridConvNet, HybridConvNetParams
from ml.models.node2vec import Node2VecModel
from ml.utils import Metric

dataset = StarWars()
data: HeteroData = dataset.data
data['Character'].timestamp_from_node = torch.full([data.num_nodes], -1)
# data['Character'].timestamp_from_node = torch.randint(-1, 7, [data.num_nodes])

train_data, val_data, test_data = EvalNodeSplitTransform()(dataset.data)
train_hdata = train_data.to_homogeneous(
    node_attrs=['timestamp_from_node'],
    edge_attrs=['timestamp_from'],
    add_node_type=False, add_edge_type=False
)

n2v_sampler = BallroomSampler(
    train_hdata.timestamp_from_node,
    train_hdata.edge_index,
    train_hdata.timestamp_from,
    window=(0, 2),
    hparams=BallroomSamplerParams(walk_length=6, walks_per_node=4, context_size=4)
)
# n2v_sampler = Node2VecSampler(
#     train_hdata.edge_index, train_hdata.num_nodes,
#     hparams=Node2VecSamplerParams(walk_length=8, walks_per_node=8, context_size=4)
# )
# uu = n2v_sampler.sample(torch.tensor([0, 1, 3, 4, 20, 42]))

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
            hparams=HybridConvNetParams(repr_dim=16, num_layers=2)
        )
        self.out_net = FCNet(16, hparams=FCNetParams(repr_dim=8, hidden_dim=[16]))

    def forward(self, batch: HeteroData) -> Tensor:
        Z_dict = self.embedder(batch)

        Z = torch.zeros(batch.batch_size, self.embedder.repr_dim)
        for store in batch.node_stores:
            node_type = store._key
            Z[store.batch_perm] = Z_dict[node_type]

        return self.out_net(Z)


model = Node2VecModel(
    embedder=HetEmbed(),
    metric=Metric.L2,
    hparams={'lr': 0.01}
)

bar = CustomProgressBar()

trainer = Trainer(gpus=None, max_epochs=50, callbacks=[bar])
trainer.fit(model, loader)

test_hgt_sampler = HGTSampler(test_data, hparams=HGTSamplerParams(num_samples=[2, 3]))
pred_data = test_hgt_sampler({
    'Character': torch.arange(test_data['Character'].num_nodes)
})
z = model(pred_data)
z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
# z = z.detach().cpu().numpy()


G, _, _, node_offsets = igraph_from_hetero(test_data)
com = G.community_multilevel()
membership = torch.tensor(com.membership, dtype=torch.long)
print(f'Number of communities: {len(com)}')

colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    '#ffd700'
]

plt.figure(figsize=(8, 8))
for i in range(len(com)):
    mask = membership == i
    plt.scatter(z[mask, 0], z[mask, 1], s=20, color=colors[i])
plt.axis('off')
plt.show()
plt.title('Louvain Labels')



test_hdata = test_data.to_homogeneous(
    node_attrs=['timestamp_from_node'],
    edge_attrs=['timestamp_from'],
    add_node_type=False, add_edge_type=False
)
temporal_index = TemporalNodeIndex().fit(
    train_hdata.timestamp_from_node,
    train_hdata.edge_index,
    train_hdata.timestamp_from
)
node_timestamps = torch.full([test_data.num_nodes], -1, dtype=torch.long)
for (node_id, t) in zip(temporal_index.node_ids, temporal_index.node_timestamps):
    if node_timestamps[int(node_id)] == -1:
        node_timestamps[int(node_id)] = t
num_timestamps = torch.max(node_timestamps) + 1


plt.figure(figsize=(8, 8))
for i in range(num_timestamps):
    mask = node_timestamps == i
    plt.scatter(z[mask, 0], z[mask, 1], s=20, color=colors[i])
plt.axis('off')
plt.title('Timestamp labels')
plt.show()
u = 0