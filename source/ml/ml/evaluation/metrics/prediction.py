from typing import Dict

from sklearn.linear_model import LogisticRegression
from torch import Tensor

from ml.utils import Metric


def link_prediction_measure(Z: Tensor, edge_index: Tensor, edge_labels: Tensor, metric: Metric = Metric.L2) -> float:
    """
    It takes the embeddings of the nodes, the edge indices, and the edge labels, and returns the accuracy of a logistic
    regression classifier that predicts the edge labels based on the distances between the embeddings

    :param metric:
    :param Z: The embeddings of the nodes
    :type Z: Tensor
    :param edge_index: The indices of the edges in the graph
    :type edge_index: Tensor
    :param edge_labels: a tensor of shape (num_edges, 1) containing the labels of the edges
    :type edge_labels: Tensor
    :return: The score of the logistic regression model.
    """
    dist = metric.pairwise_dist_fn(Z[edge_index[0, :]], Z[edge_index[1, :]]).numpy().reshape(-1, 1)
    labels = edge_labels.numpy()
    clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(dist, labels)
    score = clf.score(dist, labels)
    return score
