from collections import defaultdict
from typing import Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from tch_geometric.data import edge_type_to_str
from tch_geometric.data.subgraph import build_subgraph
from tch_geometric.loader import CustomLoader
from tch_geometric.transforms import NegativeSamplerTransform, NeighborSamplerTransform
from tch_geometric.transforms.constrastive_merge import ContrastiveMergeTransform, EdgeTypeAggregateTransform
from tch_geometric.transforms.hgt_sampling import HGTSamplerTransform, NAN_TIMESTAMP
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
from ml.models.embeddings import PinSAGEModule, HGTModule
from shared.graph import CommunityAssignment

dataset = ml.StarWars()
data = dataset[0]
data = ToUndirected(reduce='max')(data)
data = SortEdges()(data)
G = dataset.G
G.to_undirected()

repr_dim = 32
n_epochs = 20  # 10
n_comm_epochs = 10
n_clusters = 5
batch_size = 16
temporal = False


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
        edge_index_dict = {
            edge_type: self.data[edge_type].edge_index[:, idx]
            for (edge_type, idx) in zip(self.data.edge_types, idx_split)
        }

        edge_timestamp_dict = None
        if temporal:
            edge_timestamp_dict = {
                edge_type: self.data[edge_type].timestamp[idx]
                for (edge_type, idx) in zip(self.data.edge_types, idx_split)
            }

        return edge_index_dict, edge_timestamp_dict


dataset = CustomDataset(data)

neg_sampler = NegativeSamplerTransform(data, 3, 5)
# neighbor_sampler = NeighborSamplerTransform(data, [4, 3])
# neighbor_sampler = NeighborSamplerTransform(data, [3, 2])
# neighbor_sampler = HGTSamplerTransformz(data, [3, 2])
# neighbor_sampler = HGTSamplerTransform(data, [3, 2])
neighbor_sampler = HGTSamplerTransform(data, [3, 2], temporal=temporal)


def edgelist_to_subgraph(
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_timestamp_dict: Dict[EdgeType, Tensor],
):
    nodes = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    rows = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    cols = defaultdict(lambda: torch.tensor([], dtype=torch.long))
    node_counts = defaultdict(lambda: 0)

    for edge_type, edge_index in edge_index_dict.items():
        (src, _, _) = edge_type
        start = len(nodes[src])
        edge_count = edge_index.shape[1]
        nodes[src] = torch.cat([nodes[src], edge_index[0, :]])
        rows[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

        node_counts[src] += edge_count

    for edge_type, edge_index in edge_index_dict.items():
        (_, _, dst) = edge_type
        start = len(nodes[dst])
        edge_count = edge_index.shape[1]
        nodes[dst] = torch.cat([nodes[dst], edge_index[1, :]])
        cols[edge_type] = torch.arange(start, start + edge_count, dtype=torch.long)

    subgraph = build_subgraph(
        nodes, rows, cols,
        node_attrs=dict(sample_count=node_counts),
        edge_attrs=dict(timestamp=edge_timestamp_dict) if temporal else None
    )

    return subgraph
    # return nodes, (rows, cols), node_counts


class DataLoader(CustomLoader):
    def __init__(
            self,
            dataset: Dataset,
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.neg_sampler = neg_sampler
        self.neighbor_sampler = neighbor_sampler
        self.pos_neg_merge = ContrastiveMergeTransform()
        self.edge_aggr = EdgeTypeAggregateTransform()

    def sample(self, inputs):
        edge_index_dict, edge_timestamp_dict = inputs

        # Transform edges to pos subgraph
        pos_data = edgelist_to_subgraph(
            edge_index_dict,
            edge_timestamp_dict,
        )

        # Extract central `query` nodes
        ctr_nodes = {
            store._key: store.x[:store.sample_count]
            for store in pos_data.node_stores
        }

        # Sample negative nodes
        neg_data = self.neg_sampler(ctr_nodes)

        # Merge pos and neg graphs
        data = self.pos_neg_merge(pos_data, neg_data)

        # Combine unique nodes
        offset = 0
        nodes_dict = {}
        for store in data.node_stores:
            nodes, emb_mapping = torch.unique(store.x, return_inverse=True)
            nodes_dict[store._key] = nodes
            store.emb_mapping = emb_mapping + offset
            offset += len(nodes)

        # Neighbor samples
        nodes_timestamps_dict = {
            node_type: torch.full((len(idx),), NAN_TIMESTAMP, dtype=torch.int64)
            for node_type, idx in nodes_dict.items()
        }

        samples = self.neighbor_sampler(nodes_dict, nodes_timestamps_dict)

        # Aggregate pos and neg edges
        for store in data.node_stores:
            store.x = store.emb_mapping
        data_agg = self.edge_aggr(data)

        return samples, data_agg, data.node_types


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
u = 0

# embedding_module = PinSAGEModule(data.metadata(), repr_dim, normalize=False)
embedding_module = HGTModule(data.metadata(), repr_dim, repr_dim, 2, 2, use_RTE=temporal)


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

    def forward_hetero2(
            self,
            emb: Tensor,
            data: HeteroData,
    ):
        nodes = data['n'].x
        pos_edges = data[('n', 'pos', 'n')].edge_index
        neg_edges = data[('n', 'neg', 'n')].edge_index

        # Gather embeddings for edge nodes
        pc_emb = emb[nodes[pos_edges[0]], :]
        pp_emb = emb[nodes[pos_edges[1]], :]
        nc_emb = emb[nodes[neg_edges[0]], :]
        nn_emb = emb[nodes[neg_edges[1]], :]

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
        samples, pos_neg_data, node_types = batch
        emb_dict = embedding_module(samples)
        emb = torch.cat([emb_dict[node_type] for node_type in node_types], dim=0)

        posi_loss = self.posi_loss.forward_hetero2(emb, pos_neg_data)
        loss = posi_loss

        if optimize_comms:
            # Clustering Max Margin loss
            c_emb = self.centroids.weight.clone().unsqueeze(0)
            emb_q = torch.softmax(torch.sum(emb.unsqueeze(1) * c_emb, dim=2), dim=1)

            clus_loss = self.clus_loss.forward_hetero2(emb_q, pos_neg_data)
            ne = self.neg_entropy(emb_q)
            loss = posi_loss + clus_loss + ne * 0.01

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
