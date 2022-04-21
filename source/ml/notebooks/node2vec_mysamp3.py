import os.path as osp

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from sklearn.manifold import TSNE
from torch.nn import Embedding
from torch_geometric.data import HeteroData

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

from datasets import StarWars, DBLPHCN
from ml.callbacks.progress_bar import CustomProgressBar
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs, link_prediction_measure
from ml.layers.embedding import NodeEmbedding
from ml.models.node2vec import Node2VecModel
from ml.utils import Metric


def main():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
    # dataset = Planetoid(path, dataset)
    # data = dataset[0]
    dataset = StarWars()
    data = to_homogeneous(dataset[0])
    epochs = 200
    # dataset = DBLPHCN()
    # data = to_homogeneous(dataset[0])
    # epochs = 20

    pairs, labels = extract_edge_prediction_pairs(
        data.edge_index, data.num_nodes, getattr(data, f'edge_test_mask'),
        max_samples=5000
    )

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model1 = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=False).to(device)

    # embedding = Embedding(data.num_nodes, 128)
    embedding2 = NodeEmbedding(data.num_nodes, 128)
    model = Node2VecModel(
        # embedder=model1.embedding,
        # embedder=embedding,
        embedder=embedding2,
        metric=Metric.DOTP,
        hparams={'lr': 0.01}
    )

    # loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    # loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    sampler = Node2VecSampler(
        data.edge_index, data.num_nodes,
        hparams=Node2VecSamplerParams(walk_length=20, walks_per_node=10, context_size=10)
    )
    loader = NodesLoader(data.num_nodes, transform=sampler, batch_size=128, num_workers=0)

    # optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    # bar = CustomProgressBar()
    # trainer = Trainer(gpus=None, max_epochs=200, callbacks=[bar])
    # trainer.fit(model, loader)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw, node_idx in loader:
        # for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss1 = model1.loss(node_idx[pos_rw].to(device), node_idx[neg_rw].to(device))
            Z = model(node_idx)
            loss = model.loss(pos_rw.to(device), neg_rw.to(device), Z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        # z = model1()
        z = model(torch.arange(data.num_nodes).to(device))
        # acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                  z[data.test_mask], data.y[data.test_mask],
        #                  max_iter=150)
        acc = link_prediction_measure(z.detach().cpu(), pairs, labels, metric=Metric.DOTP)
        return acc

    acc = test()
    print(f'Accuracy: {acc}')

    for epoch in range(1, epochs):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())

        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, 0], z[:, 1], s=20)
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    plot_points(colors)


if __name__ == "__main__":
    main()