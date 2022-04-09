import numpy as np
from sklearn import metrics, utils, preprocessing

from ml.utils.tensor import ensure_numpy


def sim_to_sk(sim):
    if sim == 'dotp':
        return 'cosine'
    elif sim == 'cosine':
        return 'cosine'
    elif sim == 'euclidean':
        return 'euclidean'
    else:
        raise ValueError(f'Unknown similarity metric: {sim}')


def silhouette_score(X, labels, sim='dotp', sample_size=None):
    return metrics.silhouette_score(
        ensure_numpy(X), ensure_numpy(labels),
        metric=sim_to_sk(sim), sample_size=sample_size
    )


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )


def davies_bouldin_score(X, labels, sim='dotp'):
    metric = sim_to_sk(sim)
    X = ensure_numpy(X)
    labels = ensure_numpy(labels)

    X, labels = utils.check_X_y(X, labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        cluster_k = utils._safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(metrics.pairwise_distances(cluster_k, [centroid], metric=metric))

    centroid_distances = metrics.pairwise_distances(centroids, metric=metric)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)
