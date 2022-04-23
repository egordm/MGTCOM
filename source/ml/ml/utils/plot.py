from functools import lru_cache
from typing import Callable

import matplotlib as mpl
import numpy as np
import torch
from torch import Tensor

MARKER_SIZE = mpl.rcParams['lines.markersize'] ** 1.5


@lru_cache()
def create_colormap(n: int) -> np.ndarray:
    result = np.array(mpl.cm.get_cmap('tab10').colors)
    return np.concatenate([
        result,
        np.random.random((max(n - 10, 0), 3))
    ])


def draw_ellipses(ax, mus, covs, colors, **kwargs):
    for mu, cov, color in zip(mus, covs, colors):
        draw_ellipse(ax, mu, cov, color=color, **kwargs)


def draw_ellipse(ax, mu, cov, color, zorder=None, alpha=0.4, **kwargs):
    covariance = cov  # np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
    v, w = np.linalg.eigh(covariance)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(np.abs(v))
    ell = mpl.patches.Ellipse(mu, v[0], v[1], 180 + angle, color=color, **kwargs)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    if zorder:
        ell.set_zorder(zorder)

    ax.add_artist(ell)


def plot_scatter(ax, x, y, *args, markers=None, marker_idx=None, **kwargs):
    if markers is not None and marker_idx is not None:
        for i, marker in enumerate(markers):
            idx = (marker_idx == i).nonzero().reshape(-1)

            kwargs_local = kwargs.copy()
            if 'facecolors' in kwargs_local and hasattr(kwargs_local['facecolors'], '__len__'):
                kwargs_local['facecolors'] = kwargs_local['facecolors'][idx]

            if 'edgecolors' in kwargs_local and hasattr(kwargs_local['edgecolors'], '__len__'):
                kwargs_local['edgecolors'] = kwargs_local['edgecolors'][idx]

            if 'marker' in kwargs_local:
                kwargs_local.pop('marker')

            ax.scatter(x[idx], y[idx], marker=marker, *args, **kwargs_local)
    else:
        if markers is not None:
            kwargs['marker'] = markers[0]

        ax.scatter(x, y, *args, **kwargs)


def plot_decision_regions(ax, X, z, colors, cluster_fn: Callable[[Tensor], Tensor]):
    X_min, X_max = X.min(axis=0).values, X.max(axis=0).values
    arrays_for_meshgrid = [np.arange(X_min[d] - 0.1, X_max[d] + 0.1, 0.1) for d in range(X.shape[1])]
    xx, yy = np.meshgrid(*arrays_for_meshgrid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid_t = np.hstack((r1, r2))
    yhat = cluster_fn(torch.from_numpy(grid_t))
    yhat_maxed = yhat.max(dim=1).values.cpu()

    cont = ax.contourf(xx, yy, yhat_maxed.reshape(xx.shape), alpha=0.5, cmap="jet")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=colors[z],
        s=MARKER_SIZE, zorder=1
    )
    ax.set_title("Decision Boundary")
    return cont
