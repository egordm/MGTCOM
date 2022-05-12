import itertools

import numpy as np
import torch
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture
from torch_geometric.data import HeteroData, Data

from datasets import StarWars, Cora
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import community_metrics
from ml.utils.plot import create_colormap


def plot_results(X, Y_, means, covariances, index, title, colors):
    splot = plt.subplot(2, 1, 1 + index)
    _min, _max = X.min(axis=0), X.max(axis=0)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, colors)):
        if len(covar.shape) < 2:
            covar = torch.eye(32) * covar + 1e-5

        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim([float(_min[0]), float(_max[0])])
    plt.ylim([float(_min[1]), float(_max[1])])
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
# X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/StarWars/wandb/offline-run-20220501_232858-3kwsr7fr/files/embeddings_hetero.pt')
# X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/StarWars/wandb/offline-run-20220430_222415-3ftcu5qf/files/embeddings_hetero.pt')
# dataset = StarWars()
# X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/Cora/wandb/run-20220501_235544-ptk0stcg/files/embeddings_hetero.pt')
X_dict = torch.load('/data/pella/projects/University/Thesis/Thesis/source/storage/results/embedding_topo/MGCOMTopoExecutor/Cora/wandb/offline-run-20220512_191026-2gvz0k5o/files/embeddings_hetero.pt')
dataset = Cora()

X = torch.cat(list(X_dict.values()), dim=0).numpy()

# Fit a Dirichlet process Gaussian cluster_model using five components
dpgmm: BayesianGaussianMixture = mixture.BayesianGaussianMixture(
    weight_concentration_prior=10,
    # covariance_prior=0.001 * np.eye(X.shape[1]),
    weight_concentration_prior_type="dirichlet_process",
    n_components=9,
    covariance_type='full',
    # covariance_type='diag',
    # covariance_type='spherical',
    # covariance_type='tied',
    init_params="kmeans",
    max_iter=15000,
).fit(X)

plot_results(
    X,
    dpgmm.predict(X),
    dpgmm.means_,
    dpgmm.covariances_,
    1,
    "Bayesian Gaussian Mixture with a Dirichlet process prior",
    create_colormap(len(dpgmm.means_)),
)


def extract_edge_index(data):
    if isinstance(data, HeteroData):
        hdata = to_homogeneous(
            data,
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        return hdata.edge_index
    elif isinstance(data, Data):
        return data.edge_index
    else:
        return None


z = torch.from_numpy(dpgmm.predict(X))
edge_index = extract_edge_index(dataset.data)
metrics = community_metrics(z, edge_index)
print(metrics)
print(len(dpgmm.means_))

plt.show()
