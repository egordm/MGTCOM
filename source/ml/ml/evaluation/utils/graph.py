from typing import Tuple, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import negative_sampling


def extract_edge_prediction_pairs(
        edge_index: Tensor,
        num_nodes: int,
        mask: Tensor = None,
        max_samples: int = 5000
) -> Tuple[Tensor, Tensor]:
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
