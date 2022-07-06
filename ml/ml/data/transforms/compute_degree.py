import torch
from torch_geometric.data import HeteroData


def compute_degree(data: HeteroData) -> HeteroData:
    degree_dict = {
        store._key: torch.zeros(store.num_nodes, dtype=torch.long)
        for store in data.node_stores
    }

    for store in data.edge_stores:
        (_, _, dst) = store._key
        degree_dict[dst] = degree_dict[dst].scatter_add(0, store.edge_index[1], torch.ones_like(store.edge_index[1]))

    return degree_dict
