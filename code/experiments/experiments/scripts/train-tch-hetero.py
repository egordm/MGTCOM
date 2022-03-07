from collections import defaultdict
from typing import Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from tch_geometric.data import edge_type_to_str
from tch_geometric.loader import CustomLoader
from tch_geometric.transforms import NegativeSamplerTransform, NeighborSamplerTransform
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.typing import NodeType, EdgeType
from torch_scatter import scatter

import ml
from benchmarks.evaluation import get_metric_list
from experiments import HingeLoss, NegativeEntropyRegularizer
from ml import SortEdges
from ml.models.embeddings import PinSAGEModule
from shared.graph import CommunityAssignment

dataset = ml.StarWars()
data = dataset[0]
data = ToUndirected()(data)
data = SortEdges()(data)
G = dataset.G
G.to_undirected()

repr_dim = 32
n_epochs = 20  # 10
n_comm_epochs = 10
n_clusters = 5
batch_size = 24


class CustomDataset(Dataset):
    def __init__(self, data: HeteroData):
        self.data = data
        edge_borders = np.cumsum([0] + [store.num_edges for store in data.edge_stores])
        self.edge_ranges = list(zip(edge_borders[:-1], edge_borders[1:]))

    def __len__(self):
        return self.data.num_edges

    def __getitem__(self, idx):
        if not isinstance(idx, Tensor):
            idx = torch.tensor(idx, dtype=torch.long)

        # Split idx into partitions
        partitions = []
        for i, (start, end) in enumerate(self.edge_ranges):
            size = int(torch.sum((idx >= start) & (idx < end)))
            partitions.append(size)

        idx_split = torch.sort(idx).values.split(partitions)
        idx_split = [idx - start for (idx, (start, _)) in zip(idx_split, self.edge_ranges)]

        # Extract edges for each type
        result = {
            edge_type: self.data[edge_type].edge_index[:, idx]
            for (edge_type, idx) in zip(self.data.edge_types, idx_split)
        }

        return result


dataset = CustomDataset(data)

neg_sampler = NegativeSamplerTransform(data, 3, 5)
# neighbor_sampler = NeighborSamplerTransform(data, [4, 3])
neighbor_sampler = NeighborSamplerTransform(data, [3, 2])


def edgelist_to_subgraph(edge_indexes: Dict[EdgeType, Tensor]):
    nodes = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    rows = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    cols = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    node_counts = defaultdict(lambda: 0)

    for edge_type, edge_index in edge_indexes.items():
        (src, _, _) = edge_type
        start = len(nodes[src])
        edge_count = edge_index.shape[1]
        nodes[src] = torch.cat([nodes[src], edge_index[0, :]])
        rows[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

        node_counts[src] += edge_count

    for edge_type, edge_index in edge_indexes.items():
        (_, _, dst) = edge_type
        start = len(nodes[dst])
        edge_count = edge_index.shape[1]
        nodes[dst] = torch.cat([nodes[dst], edge_index[1, :]])
        cols[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

    return nodes, (rows, cols), node_counts


class DataLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.neg_sampler = neg_sampler
        self.neighbor_sampler = neighbor_sampler

    def sample(self, inputs):
        # Edge rel to type mapping
        edge_rel_to_type = {
            edge_type_to_str(edge_type): edge_type
            for edge_type in self.dataset.data.edge_types
        }

        # Positive samples
        # TODO: use positive sampler instead
        pos_nodes, pos_edges, node_counts = edgelist_to_subgraph(inputs)
        pos_edges = {
            rel: (pos_edges[0][rel], pos_edges[1][rel])
            for rel in pos_edges[0].keys()
        }

        # Extract center nodes
        ctr_nodes = {
            node_type: pos_nodes[node_type][:count]
            for node_type, count in node_counts.items()
        }

        # Negative samples
        neg_nodes, neg_edges, _ = self.neg_sampler(ctr_nodes)
        neg_edges = {
            edge_rel_to_type[rel]: (neg_edges[0][rel], neg_edges[1][rel])
            for rel in neg_edges[0].keys()
        }

        # Merge nodes and correct edge indices
        for rel, edges in neg_edges.items():
            (src, _, dst) = rel
            neg_edges[rel] = (edges[0], edges[1] + len(pos_nodes[dst]))

        nodes = defaultdict(lambda: torch.tensor([], dtype=torch.long))
        for node_type, nodes_list in pos_nodes.items():
            nodes[node_type] = nodes_list
        for node_type, nodes_list in neg_nodes.items():
            nodes[node_type] = torch.cat([nodes[node_type], nodes_list])

        # Prune nodes
        nodes_inv = {}
        for node_type, node_list in nodes.items():
            node_list, inverse_indices = torch.unique(
                node_list, return_inverse=True,
            )
            nodes[node_type] = node_list
            nodes_inv[node_type] = inverse_indices

        # Neighbor samples
        samples = self.neighbor_sampler(nodes)
        for store in samples.edge_stores:
            store.edge_weight = torch.ones(store.num_edges, dtype=torch.float)

        return samples, nodes_inv, pos_edges, neg_edges


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
u = 0

embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)


class HingeLoss(torch.nn.Module):
    def __init__(self, delta=1.0, agg_pos='mean', agg_neg='max') -> None:
        super().__init__()
        self.delta = delta
        self.agg_pos = agg_pos
        self.agg_neg = agg_neg

    def forward(
            self,
            emb_src: Tensor,
            emb_dst: Tensor,
            inv_idx: Tensor,
            pos_edges: Tuple[Tensor, Tensor],
            neg_edges: Tuple[Tensor, Tensor],
    ):
        # Gather embeddings for edge nodes
        pc_emb = emb_src[inv_idx[pos_edges[0]], :]
        pp_emb = emb_dst[inv_idx[pos_edges[1]], :]
        nc_emb = emb_src[inv_idx[neg_edges[0]], :]
        nn_emb = emb_dst[inv_idx[neg_edges[1]], :]

        # Compute positive and negative distances
        p_d_full = torch.bmm(pc_emb.unsqueeze(1), pp_emb.unsqueeze(2)).view(-1)
        p_d = scatter(p_d_full, pos_edges[0], dim=0, reduce=self.agg_pos)

        n_d_full = torch.bmm(nc_emb.unsqueeze(1), nn_emb.unsqueeze(2)).view(-1)
        n_d = scatter(n_d_full, neg_edges[0], dim=0, reduce=self.agg_neg)

        # Compute loss
        loss = torch.clip(n_d - p_d + self.delta, min=0).mean()

        return loss

    def forward_hetero(
            self,
            emb: Dict[NodeType, Tensor],
            inv_idx: Dict[NodeType, Tensor],
            pos_edges: Dict[EdgeType, Tuple[Tensor, Tensor]],
            neg_edges: Dict[EdgeType, Tuple[Tensor, Tensor]],
    ):
        pc_emb_l = []
        pp_emb_l = []
        nc_emb_l = []
        nn_emb_l = []
        pos_edges_idx_l = []
        neg_edges_idx_l = []

        for edge_type in data.edge_types:
            if edge_type not in pos_edges:
                continue

            (src, _, dst) = edge_type
            emb_src = emb[src]
            emb_dst = emb[dst]
            l_inv_idx = inv_idx[src]
            l_pos_edges = pos_edges[edge_type]
            l_neg_edges = neg_edges[edge_type]

            pc_emb = emb_src[l_inv_idx[l_pos_edges[0]], :]
            pp_emb = emb_dst[l_inv_idx[l_pos_edges[1]], :]
            nc_emb = emb_src[l_inv_idx[l_neg_edges[0]], :]
            nn_emb = emb_dst[l_inv_idx[l_neg_edges[1]], :]

            pc_emb_l.append(pc_emb)
            pp_emb_l.append(pp_emb)
            nc_emb_l.append(nc_emb)
            nn_emb_l.append(nn_emb)
            pos_edges_idx_l.append(l_pos_edges[0])
            neg_edges_idx_l.append(l_neg_edges[0])

        pc_emb = torch.cat(pc_emb_l, dim=0)
        pp_emb = torch.cat(pp_emb_l, dim=0)
        nc_emb = torch.cat(nc_emb_l, dim=0)
        nn_emb = torch.cat(nn_emb_l, dim=0)
        pos_edges_idx = torch.cat(pos_edges_idx_l, dim=0)
        neg_edges_idx = torch.cat(neg_edges_idx_l, dim=0)

        # Compute positive and negative distances
        p_d_full = torch.bmm(pc_emb.unsqueeze(1), pp_emb.unsqueeze(2)).view(-1)
        p_d = scatter(p_d_full, pos_edges_idx, dim=0, reduce=self.agg_pos)

        n_d_full = torch.bmm(nc_emb.unsqueeze(1), nn_emb.unsqueeze(2)).view(-1)
        n_d = scatter(n_d_full, neg_edges_idx, dim=0, reduce=self.agg_neg)

        # Compute loss
        loss = torch.clip(n_d - p_d + self.delta, min=0).mean()

        return loss


class MainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.centroids = torch.nn.Embedding(n_clusters, repr_dim)
        self.posi_loss = HingeLoss()
        self.clus_loss = HingeLoss()
        self.neg_entropy = NegativeEntropyRegularizer()

    def forward(self, batch, optimize_comms=False):
        samples, inv_idx, pos_edges, neg_edges = batch
        emb = embedding_module(samples)

        posi_loss = self.posi_loss.forward_hetero(emb, inv_idx, pos_edges, neg_edges)
        loss = posi_loss

        if optimize_comms:
            # Clustering Max Margin loss
            c_emb = self.centroids.weight.clone().unsqueeze(0)
            emb_q = {
                node_type: torch.softmax(torch.sum(emb_l.unsqueeze(1) * c_emb, dim=2), dim=1)
                for node_type, emb_l in emb.items()
            }

            clus_loss = self.clus_loss.forward_hetero(emb_q, inv_idx, pos_edges, neg_edges)
            # ne = self.neg_entropy(emb_q)
            loss = posi_loss + clus_loss  # + ne * 0.01

        return loss


model = MainModel()
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


def train(epoch):
    model.train()
    total_loss = 0
    optimize_comms = epoch > n_epochs
    for batch in data_loader:
        optimizer.zero_grad()

        loss = model(batch, optimize_comms)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


for epoch in range(0, n_epochs + n_comm_epochs):
    # if epoch == comm_epoch and initialize is not None:
    #     embeddings = get_embeddings()
    #     c = initial_clustering(embeddings)
    #     model.centroids = torch.nn.Embedding.from_pretrained(c, freeze=False)

    loss = train(epoch)
    # acc = test()
    acc = np.nan
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')  #


def get_embeddings():
    model.eval()
    inputs = {
        'Character': torch.arange(0, data['Character'].num_nodes),
    }
    samples = neighbor_sampler(inputs)
    emb = embedding_module(samples)
    return emb['Character'].detach()


embeddings = get_embeddings()
print('Reusing trained centers')
centers = model.centroids.weight.detach()
q = torch.softmax(torch.mm(embeddings, centers.transpose(1, 0)), dim=-1)
I = q.argmax(dim=-1)
I = I.numpy()

labeling = pd.Series(I.squeeze(), index=range(len(I)), name="cid")
labeling.index.name = "nid"
comlist = CommunityAssignment(labeling)

metrics = get_metric_list(ground_truth=False, overlapping=False)
results = pd.DataFrame([
    {
        'metric': metric_cls.metric_name(),
        'value': metric_cls.calculate(G, comlist)
    }
    for metric_cls in metrics]
)
print(results)
