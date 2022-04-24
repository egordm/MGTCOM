from collections import defaultdict
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from ml.utils import EPS


def newman_girvan_modularity(
        edge_index: Tensor,
        clustering: Tensor,
        num_clusters: int = None,
) -> float:
    """
    It calculates the modularity of a graph given a clustering.

    Difference the fraction of intra community edges of a partition with the expected number of such edges if
    distributed according to a null model.

    Generally higher is better.

    :param edge_index: The edge list of the graph
    :type edge_index: Tensor
    :param clustering: The clustering of the graph
    :type clustering: Tensor
    :param num_clusters: The number of clusters in the graph
    :type num_clusters: int
    :return: The modularity of the graph.
    """

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

    if not num_clusters:
        num_clusters = clustering.max().item() + 1

    # Calculate node degrees
    degree = torch.zeros(num_nodes, dtype=torch.long)
    (row, col) = edge_index
    degree = degree.scatter_add(0, row, torch.ones_like(row))
    degree = degree.scatter_add(0, col, torch.ones_like(col))

    # Calculate the modularity per community
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


def conductance(
        edge_index: Tensor,
        clustering: Tensor,
        num_clusters: int = None,
) -> float:
    """
    It calculates the conductance of a graph given its edge index and clustering.

    Conductance: Fraction of total edge volume that points outside the community.
    Generally lower is better.

    :param edge_index: The edge index of the graph
    :type edge_index: Tensor
    :param clustering: The clustering of the graph
    :type clustering: Tensor
    :param num_clusters: The number of clusters to use
    :type num_clusters: int
    :return: The mean conductance of the graph.
    """

    # Calculate number of edges
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        raise ValueError("A graph without link has an undefined conductance")

    num_nodes = len(clustering)
    assert num_nodes >= edge_index.max().item() + 1, "Edge index contains more nodes than clustering"

    if not num_clusters:
        num_clusters = clustering.max().item() + 1

    (row, col) = edge_index
    src_comms, dst_comms = clustering[row], clustering[col]

    # Count the number of inner edges per community
    mask = (src_comms == dst_comms).nonzero().squeeze()
    inner_edges = torch.zeros(num_clusters).scatter_add_(0, src_comms[mask], torch.ones_like(mask, dtype=torch.float))

    # Count number of outer edges per community
    mask = (src_comms != dst_comms).nonzero().squeeze()
    outer_edges = torch.zeros(num_clusters).scatter_add_(0, src_comms[mask], torch.ones_like(mask, dtype=torch.float))

    # Calculate conductance per community
    con_c = outer_edges / ((2 * inner_edges) + outer_edges + EPS)

    return con_c.mean()
