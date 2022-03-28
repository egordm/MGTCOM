from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

Weight = float


@dataclass
class CooGraph:
    n_nodes: int
    edge_index: Tensor
    weights: Tensor
    edge_index_rev: Tensor
    perm_rev: Tensor

    @staticmethod
    def from_edgelist(n_nodes, edge_index: Tensor, weights: Tensor) -> 'CooGraph':
        edge_index_rev = edge_index.flip(0)
        edge_index_rev, perm = edge_index_rev.unique(dim=1, sorted=True, return_inverse=True)
        perm_rev = torch.zeros(len(perm), dtype=torch.long)
        perm_rev[perm] = torch.arange(len(perm), dtype=torch.long)
        return CooGraph(n_nodes, edge_index, weights, edge_index_rev, perm_rev)

    def out_edges(self, node: int) -> Tuple[Tensor, Tensor]:
        rows = self.edge_index[0]
        e_from, e_to = torch.searchsorted(rows, torch.tensor([node, node + 1], dtype=torch.long))
        perm = torch.arange(e_from, e_to, dtype=torch.long)
        return self.edge_index[1, e_from:e_to], perm

    def in_edges(self, node: int) -> Tuple[Tensor, Tensor]:
        rows = self.edge_index_rev[0]
        e_from, e_to = torch.searchsorted(rows, torch.tensor([node, node + 1], dtype=torch.long))
        perm = self.perm_rev[torch.arange(e_from, e_to, dtype=torch.long)]
        return self.edge_index_rev[1, e_from:e_to], perm

    def incident(self, node: int, undirected: bool = True) -> Tuple[Tensor, Tensor]:
        if undirected:
            edge_index_out, perm_out = self.out_edges(node)
            edge_index_in, perm_in = self.in_edges(node)
            edge_index = torch.cat([edge_index_out, edge_index_in], dim=0)
            perm = torch.cat([perm_out, perm_in], dim=0)
            return edge_index, perm
        else:
            raise NotImplementedError


@dataclass
class MultilevelCommunity:
    size: int  # Size of the community
    weight_inside: Weight  # Sum of edge weights inside community
    weight_all: Weight  # Sum of edge weights starting/ending in the community


@dataclass
class MultilevelCommunityList:
    n_comms: int  # Number of communities
    n_vertices: int  # Number of vertices
    weight_sum: Weight  # Sum of edges weight in the whole graph
    communities: List[MultilevelCommunity]  # List of communities
    membership: Tensor  # Community IDs

    def modularity(self, resolution: float) -> float:
        """
        Computes the modularity of a community partitioning
        """
        result = 0.0
        m = self.weight_sum

        for i in range(0, self.n_vertices):
            comm = self.communities[i]
            if comm.size > 0:
                result += (comm.weight_inside - resolution * comm.weight_all * comm.weight_all / m) / m

        return float(result)


def multilevel_simplify_multiple(edge_index: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Merges multiple edges and returns new edge id's for each edge in |E|log|E|
    """
    edge_index, perm = torch.unique(edge_index, dim=1, return_inverse=True, sorted=True)
    # Recalculate edge weights
    weights = torch.zeros(edge_index.shape[1], dtype=torch.float).index_add(0, perm, weights)
    return edge_index, weights, perm


def multilevel_community_modularity_gain(
        comms: MultilevelCommunityList,
        cid: int, _vertex: int,
        weight_all: float, weight_inside: float,
        resolution: float
) -> float:
    return weight_inside - resolution * comms.communities[cid].weight_all * weight_all / comms.weight_sum


def multilevel_shrink(edge_index: Tensor, membership: Tensor) -> [int, Tensor]:
    """
    Shrinks communities into single vertices, keeping all the edges.
    """
    n_edges = edge_index.shape[1]
    edge_index_new = membership[torch.cat([edge_index[0, :], edge_index[1, :]], dim=0)]
    nodes, perm = torch.unique(edge_index_new, return_inverse=True, sorted=True)
    return len(nodes), torch.vstack([
        perm[0:n_edges], perm[n_edges:]
    ])


def reindex_membership(membership: Tensor) -> Tensor:
    comms, perm = torch.unique(membership, return_inverse=True)
    return len(comms), perm


def multilevel_community_links(
        graph: CooGraph,
        comms: MultilevelCommunityList,
        vertex: int
):
    """
    Given a graph, a community structure and a vertex ID, this method calculates:
    - links: the list of edge IDs that are incident on the vertex
    - weight_all: the total weight of these edges
    - weight_inside: the total weight of edges that stay within the same community where the given vertex is right now,
        excluding loop edges
    - weight_loop: the total weight of loop edges
    - c_ids and c_weights: together these two vectors list the communities incident on this vertex and the total
        weight of edges pointing to these communities
    """
    weight_all, weight_inside, weight_loop = (0.0, 0.0, 0.0)

    # Get the list of incident edges
    cid = comms.membership[vertex]
    neighbors, neigh_edge_ids = graph.incident(vertex)

    n_edges = len(neigh_edge_ids)
    link_cid = torch.zeros(n_edges, dtype=torch.long)
    link_weight = torch.zeros(n_edges, dtype=torch.float)

    weight = graph.weights[neigh_edge_ids]
    to = neighbors
    weight_all += weight.sum()

    # Self loops
    self_loop_mask = (to == vertex).nonzero()
    weight_loop += weight[self_loop_mask].sum()
    link_cid[self_loop_mask] = cid
    link_weight[self_loop_mask] = 0

    # Normal Edges
    to_cid = comms.membership[to]
    weight_inside += weight[torch.logical_and(to != vertex, to_cid == cid)].sum()
    edge_mask = (to != vertex).nonzero()
    link_cid[edge_mask] = to_cid[edge_mask]
    link_weight[edge_mask] = weight[edge_mask]

    # Sort links by community ID and merge the same
    c_ids, link_perm = torch.unique(link_cid, return_inverse=True, sorted=True)
    c_weights = torch.zeros(c_ids.shape[0], dtype=torch.float).index_add(0, link_perm, link_weight)

    return (
        float(weight_all), float(weight_inside), float(weight_loop),
        c_ids, c_weights
    )


def community_multilevel_step(
        graph: CooGraph,
        resolution: float = 1.0,
) -> Tuple[CooGraph, Tensor, float]:
    """
    Performs a single step of the multi-level modularity optimization method
    :param graph: The input graph. It must be an undirected graph.
    :param resolution: Resolution parameter. Must be greater than or equal to 0. Default is 1.
    :return:
    """

    # Initialize list of communities from graph vertices
    n_nodes = graph.n_nodes
    node_order = torch.arange(0, n_nodes, dtype=torch.long)
    comms = MultilevelCommunityList(
        n_comms=n_nodes, n_vertices=n_nodes,
        weight_sum=2 * graph.weights.sum(),
        membership=torch.arange(0, n_nodes, dtype=torch.long),
        communities=[
            MultilevelCommunity(1, 0, 0)
            for _ in range(n_nodes)
        ]
    )

    for ((ffrom, fto), w) in zip(graph.edge_index.t(), graph.weights):  # TODO: Can be sped up
        comms.communities[ffrom].weight_all += w
        comms.communities[fto].weight_all += w
        if ffrom == fto:
            comms.communities[ffrom].weight_inside += 2 * w

    q = comms.modularity(resolution)
    pass_g = 1

    while True: # Pass begin
        pass_q = q
        changed = 0

        # Save the current membership, it will be restored in case of worse result
        tmp_comm = comms.n_comms, comms.membership.clone()

        for i in range(0, n_nodes):
            # Exclude vertex from its current community
            ni = int(node_order[i])
            weight_all, weight_inside, weight_loop, c_ids, c_weights = multilevel_community_links(graph, comms, ni)

            old_id = int(comms.membership[ni])
            new_id = int(old_id)

            # Update old community
            comms.membership[ni] = -1
            comms.communities[old_id].size -= 1
            if comms.communities[old_id].size == 0:
                comms.n_comms -= 1
            comms.communities[old_id].weight_all -= weight_all
            comms.communities[old_id].weight_inside -= 2 * weight_inside + weight_loop

            # Find new community to join with the best modification gain
            max_q_gain, max_weight = 0, weight_inside
            for (ci, cw) in zip(c_ids, c_weights):
                q_gain = multilevel_community_modularity_gain(comms, ci, ni, weight_all, cw, resolution)

                if q_gain > max_q_gain:
                    new_id = int(ci)
                    max_q_gain = float(q_gain)
                    max_weight = float(cw)

            # Add vertex to "new" community and update it
            comms.membership[ni] = new_id
            if comms.communities[new_id].size == 0:
                comms.n_comms += 1
            comms.communities[new_id].size += 1
            comms.communities[new_id].weight_all += weight_all
            comms.communities[new_id].weight_inside += 2 * max_weight + weight_loop

            if new_id != old_id:
                changed += 1

        q = comms.modularity(resolution)
        if changed and q > pass_q:
            pass_g += 1
        else:
            # No changes or the modularity became worse, restore last membership
            comms.n_comms, comms.membership = tmp_comm
            break

    modularity = q

    # Shrink the nodes of the graph according to the present community structure and simplify the resulting graph
    n_nodes, membership = reindex_membership(comms.membership)
    edge_index = torch.vstack([
        membership[graph.edge_index[0, :]],
        membership[graph.edge_index[1, :]],
    ])

    # Update edge weights after shrinking and simplification
    # Here we reuse the edges vector as we don't need the previous contents anymore
    edge_index, weights, edge_perm = multilevel_simplify_multiple(edge_index, graph.weights)
    new_graph = CooGraph.from_edgelist(n_nodes, edge_index, weights)

    return new_graph, membership, modularity


def louvain(
        n_count: int,
        edge_index: Tensor,
        weights: Tensor,
        resolution: float = 1.0,
):
    """
    :param resolution: Resolution parameter. Must be greater than or equal to 0. Default is 1.
    """
    # Sort edge index
    edge_index, weights, _ = multilevel_simplify_multiple(edge_index, weights)
    graph = CooGraph.from_edgelist(n_count, edge_index, weights)

    level = 1
    prev_q, q = -1, -1
    level_membership = torch.arange(0, n_count, dtype=torch.long)
    modularity = []
    level_memberships = []

    step_graph = graph
    while True:
        # Remember the previous modularity and vertex count, do a single step
        prev_q, prev_n_count = (q, step_graph.n_nodes)
        step_graph, m, q = community_multilevel_step(step_graph, resolution)

        if prev_n_count == step_graph.n_nodes or q < prev_q:
            break

        level_membership = m[level_membership]
        modularity.append(q)
        level_memberships.append(level_membership)

        level += 1

    if len(modularity) == 0:
        # TODO: compute modularity
        q = -1
        modularity.append(q)

    return level_memberships, modularity
