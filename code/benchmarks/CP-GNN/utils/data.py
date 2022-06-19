import os
import pickle
from pathlib import Path

import torch
import dgl
from sklearn.model_selection import train_test_split

from .preprocess import EdgesDataset
from typing import Tuple, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import negative_sampling

EdgePredictionBatch = Tuple[Tensor, Tensor]

ROOT_PATH = Path(__file__).parent.parent.parent.parent

CACHE_PATH = ROOT_PATH / 'storage/cache/baselines/CP-GNN'


class ExtGraphDataLoader(object):
    def __init__(self, dataset: str, dataset_version: str, data_config) -> None:
        super().__init__()
        path = ROOT_PATH / 'storage/exports' / dataset / dataset_version
        self.data = torch.load(path / "train.pt")
        self.data_config = data_config

        print(self.data)

        self.heter_graph = dgl.heterograph(
            data_dict={
                k: (v[0], v[1])
                for k, v in self.data.edge_index_dict.items()
            },
            num_nodes_dict=self.data.num_nodes_dict,
        )

        self.k_hop_graph_path = CACHE_PATH / dataset / dataset_version
        self.k_hop_graph_path.mkdir(parents=True, exist_ok=True)

    def load_k_hop_train_data(self):
        pass

    def _load_k_hop_graph(self, hg, k, primary_type):
        print(f'Process: {k} hop graph')
        k_hop_graph_path = self.k_hop_graph_path / f'{primary_type}_{k}_hop_graph.pkl'
        if not k_hop_graph_path.exists():
            ntype = hg.ntypes
            primary_type_id = ntype.index(primary_type)
            homo_g = dgl.to_homo(hg)

            p_nodes_id = homo_g.filter_nodes(
                lambda nodes: (nodes.data['_TYPE'] == primary_type_id))  # Find the primary nodes ID
            min_p = torch.min(p_nodes_id).item()
            max_p = torch.max(p_nodes_id).item()

            raw_adj = homo_g.adjacency_matrix_scipy()
            adj_k = raw_adj ** k
            p_adj = adj_k[min_p:max_p, min_p:max_p]

            row, col = p_adj.nonzero()
            p_g = dgl.graph((row, col))
            with k_hop_graph_path.open('wb') as f:
                pickle.dump(p_g, f, protocol=4)
        else:
            with k_hop_graph_path.open('rb') as f:
                p_g = pickle.load(f)

        return p_g

    def load_train_k_context_edges(self, hg, K, primary_type, pos_num_for_each_hop, neg_num_for_each_hop):
        edges_data_dict = {}
        for k in range(1, K + 2):
            k_hop_primary_graph = self._load_k_hop_graph(hg, k, primary_type)
            k_hop_edge = EdgesDataset(k_hop_primary_graph, pos_num_for_each_hop[k], neg_num_for_each_hop[k])
            edges_data_dict[k] = k_hop_edge
        return edges_data_dict


def extract_edge_prediction_pairs(
    edge_index: Tensor,
    num_nodes: int,
    mask: Tensor = None,
    max_samples: int = 5000
) -> EdgePredictionBatch:
    """
    It takes an edge index and a mask, and returns a list of positive and negative pairs, along with a list of labels

    :param edge_index: The edge indices of the graph
    :type edge_index: Tensor
    :param num_nodes: The number of nodes in the graph
    :type num_nodes: int
    :param mask: a boolean mask that indicates which edges to use for training
    :type mask: Tensor
    :param max_samples: The maximum number of positive samples to use, defaults to 5000
    :type max_samples: int (optional)
    :return: A tuple of two tensors. The first tensor is a concatenation of the positive and negative pairs. The second
    tensor is a concatenation of the labels for the positive and negative pairs.
    """
    if mask is None:
        mask = torch.ones_like(edge_index.shape[1], dtype=torch.bool)

    pos_pairs = torch.unique(edge_index[:, mask].t(), dim=0).t()
    if pos_pairs.shape[1] > max_samples:
        pos_pairs = pos_pairs[:, torch.randperm(pos_pairs.shape[1])[:max_samples]]

    neg_pairs = negative_sampling(edge_index, num_nodes, num_neg_samples=pos_pairs.shape[1])

    pairs = torch.cat([pos_pairs, neg_pairs], dim=1)
    labels = torch.cat([torch.ones(pos_pairs.shape[1]), torch.zeros(neg_pairs.shape[1])])

    return pairs, labels
