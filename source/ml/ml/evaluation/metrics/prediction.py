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
        metric: Metric = Metric.DOTP
) -> Tuple[float, Dict[str, float]]:
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
    :return: The accuracy and the ROC AUC of the logistic regression classifier
    """
    X = metric.pairwise_dist_fn(Z[edge_index[0, :]], Z[edge_index[1, :]]).reshape(-1, 1)
    y = edge_labels

    return prediction_measure(X, y)


def prediction_measure(
        X: Tensor,
        y: Tensor
) -> Tuple[float, Dict[str, float]]:
    """
    It fits a logistic regression model to the data and returns the accuracy and ROC AUC

    :param X: Tensor - the input data
    :type X: Tensor
    :param y: The target variable
    :type y: Tensor
    :return: The accuracy and the roc_auc
    """
    is_multiclass = y.max() > 1

    X = X.numpy()
    X = StandardScaler().fit_transform(X)
    y = y.numpy()
    clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X, y)
    # clf = SVC().fit(X, y)
    probs = clf.predict_proba(X)
    y_hat = probs.argmax(axis=1)

    accuracy = clf.score(X, y)
    metrics = {
        'accuracy': accuracy
    }

    if is_multiclass:
        metrics['f1_macro'] = f1_score(y, y_hat, average='macro')
        metrics['f1_micro'] = f1_score(y, y_hat, average='micro')
    else:
        metrics['roc_auc'] = roc_auc_score(y, probs[:, 1])
        metrics['f1'] = f1_score(y, y_hat)

    return accuracy, metrics
