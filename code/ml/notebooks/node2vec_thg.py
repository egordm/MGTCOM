import os.path as osp

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.nn import Node2Vec

from datasets import StarWars, DBLPHCN
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs, link_prediction_measure
from ml.utils import Metric


def main():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
    # dataset = Planetoid(path, dataset)
    # data = dataset[0]
    dataset = StarWars()
    data = to_homogeneous(dataset[0])
    epochs = 200
    dataset = DBLPHCN()
    data = to_homogeneous(dataset[0])
    epochs = 20

    pairs, labels = extract_edge_prediction_pairs(
        data.edge_index, data.num_nodes, getattr(data, f'edge_val_mask'),
        max_samples=5000
    )

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=False).to(device)

    # loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    # optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        # acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                  z[data.test_mask], data.y[data.test_mask],
        #                  max_iter=150)
        acc = link_prediction_measure(z.detach().cpu(), pairs, labels, metric=Metric.DOTP)
        return acc

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