from collections import defaultdict
from typing import Dict, NamedTuple, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor

from ml.utils import Metric


def link_prediction_measure(
        Z: Tensor,
        edge_index: Tensor,
        edge_labels: Tensor,
        metric: Metric = Metric.L2,
        max_iter: int = 100,
) -> Tuple[float, Dict[str, float]]:
    """
    It takes the embeddings of the nodes, the edge indices, and the edge labels, and returns the accuracy of a logistic
    regression classifier that predicts the edge labels based on the distances between the embeddings

    :param max_iter:
    :param metric:
    :param Z: The embeddings of the nodes
    :type Z: Tensor
    :param edge_index: The indices of the edges in the graph
    :type edge_index: Tensor
    :param edge_labels: a tensor of shape (num_edges, 1) containing the labels of the edges
    :type edge_labels: Tensor
    :return: The accuracy and the ROC AUC of the logistic regression classifier
    """
    X = metric.pairwise_dist_fn(Z[edge_index[0, :]], Z[edge_index[1, :]]).reshape(-1, 1)
    y = edge_labels

    return prediction_measure(X, y, max_iter=max_iter)


def prediction_measure(
        X: Tensor,
        y: Tensor,
        max_iter: int = 100,
        n_fits: int = 3,
) -> Tuple[float, Dict[str, float]]:
    """
    It fits a logistic regression model to the data and returns the accuracy and ROC AUC

    :param X: Tensor - the input data
    :type X: Tensor
    :param y: The target variable
    :type y: Tensor
    :param max_iter: The maximum number of iterations for the logistic regression
    :type max_iter: int
    :param n_fits: The number of times to fit the model
    :type n_fits: int
    :return: The accuracy and the roc_auc
    """
    is_multiclass = y.max() > 1

    X = X.numpy()
    X = StandardScaler().fit_transform(X)
    y = y.numpy()
    metrics = defaultdict(lambda: 0)

    for _ in range(n_fits):
        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=max_iter, n_jobs=-1).fit(X, y)
        # clf = LogisticRegression(solver='saga', multi_class='auto', max_iter=max_iter).fit(X, y)
        probs = clf.predict_proba(X)
        y_hat = probs.argmax(axis=1)

        accuracy = clf.score(X, y)
        metrics['accuracy'] += accuracy / n_fits

        if is_multiclass:
            metrics['f1_macro'] += f1_score(y, y_hat, average='macro') / n_fits
            metrics['f1_micro'] += f1_score(y, y_hat, average='micro') / n_fits
        else:
            metrics['roc_auc'] += roc_auc_score(y, probs[:, 1]) / n_fits
            metrics['f1'] += f1_score(y, y_hat) / n_fits

    return metrics['accuracy'], metrics
