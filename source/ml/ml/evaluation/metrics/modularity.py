from collections import defaultdict
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


def newman_girvan_modularity_hetero(
        data: HeteroData,
        clustering: Dict[NodeType, Tensor],
        num_clusters: int = None,
) -> float:
    # Create a undirected list of unique edges
    edges = defaultdict(list)
    for store in data.edge_stores:
        (src, _, dst) = store._key
        edge_index = store.edge_index.t()
        if src == dst:
            edge_index = torch.sort(edge_index, dim=1).values

        perm = torch.sort(edge_index[:, 0] * store.size()[0] + edge_index[:, 1]).indices
        edge_index = edge_index[perm, :]

        edges[(src, dst)].append(edge_index)

    edges = {k: torch.unique(torch.cat(v, dim=0), dim=0).t() for k, v in edges.items()}

    # Calculate number of edges
    num_edges = sum(v.shape[1] for v in edges.values())
    if num_edges == 0:
        raise ValueError("A graph without link has an undefined modularity")

    # Calculate node degrees
    degree = {
        store._key: torch.zeros(store.num_nodes, dtype=torch.long)
        for store in data.node_stores
    }

    for (src, dst), (row, col) in edges.items():
        degree[src] = degree[src].scatter_add(0, row, torch.ones_like(row))
        degree[dst] = degree[dst].scatter_add(0, col, torch.ones_like(col))

    # Calculate the modularity per community
    if not num_clusters:
        num_clusters = max([v.max() for v in clustering.values()]) + 1

    inc = torch.zeros(num_clusters)
    for (src, dst), (row, col) in edges.items():
        src_comms = clustering[src][row]
        dst_comms = clustering[dst][col]
        mask = (src_comms == dst_comms).nonzero().squeeze()

        weights = torch.full([len(mask)], 1.0, dtype=torch.float)
        weights[row[mask] == col[mask]] = 2.0

        inc = inc.scatter_add_(0, src_comms[mask], weights)

    deg = torch.zeros(num_clusters, dtype=torch.long)
    for node_type, degrees in degree.items():
        deg = deg.scatter_add_(0, clustering[node_type], degrees)

    # Aggregate the modularity
    mod = (inc / num_edges) - (deg / (2 * num_edges)) ** 2
    return mod.sum()


def newman_girvan_modularity(
        edge_index: Tensor,
        clustering: Tensor,
        num_clusters: int = None,
) -> float:
    # Create a undirected list of unique edges
    edge_index = torch.unique(torch.cat([
        edge_index.t(),
        torch.flip(edge_index, dims=[0]).t()
    ]), dim=0, sorted=True).t()

    # Calculate number of edges
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        raise ValueError("A graph without link has an undefined modularity")

    num_nodes = len(clustering)
    assert num_nodes >= edge_index.max().item() + 1, "Edge index contains more nodes than clustering"

    # Calculate node degrees
    degree = torch.zeros(num_nodes, dtype=torch.long)
    (row, col) = edge_index
    degree = degree.scatter_add(0, row, torch.ones_like(row))
    degree = degree.scatter_add(0, col, torch.ones_like(col))

    # Calculate the modularity per community
    if not num_clusters:
        num_clusters = clustering.max().item() + 1

    src_comms, dst_comms = clustering[row], clustering[col]
    mask = (src_comms == dst_comms).nonzero().squeeze()

    weights = torch.full([len(mask)], 1.0, dtype=torch.float)
    weights[row[mask] == col[mask]] = 2.0
    inc = torch.zeros(num_clusters).scatter_add_(0, src_comms[mask], weights)

    deg = torch.zeros(num_clusters, dtype=torch.long)
    deg = deg.scatter_add_(0, clustering, degree)

    # Aggregate the modularity
    mod = (inc / num_edges) - (deg / (2 * num_edges)) ** 2
    return mod.sum()
