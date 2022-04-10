from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import umap.plot
from torch import Tensor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from ml.data.transforms.mapping import PCAMapper, mapper_cls
from ml.data.transforms.subsampling import Subsampler
from ml.models.dpm_clustering import DPMClusteringModel
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class GMMVisualizerCallback(Callback):
    subsampler: Subsampler
    mapper: PCAMapper
    X_t: Tensor

    def __init__(self, logging_interval: int = 3, max_points: int = 1000, mapper='pca') -> None:
        super().__init__()
        self.logging_interval = logging_interval
        self.max_points = max_points
        self.enabled = True
        self.cmap = mpl.cm.get_cmap('tab20')
        self.marker_size = mpl.rcParams['lines.markersize'] ** 2
        self.mapper_type = mapper

    def setup(self, trainer: Trainer, pl_module: DPMClusteringModel, stage: Optional[str] = None) -> None:
        self.X = pl_module.dataset[torch.arange(len(pl_module.dataset), dtype=torch.long)].cpu()
        self.mapper = mapper_cls(self.mapper_type if self.X.shape[1] != 2 else 'none')(n_components=2)
        logger.info('Fitting mapper')
        self.mapper.fit(self.X)

        self.subsampler = Subsampler(max_points=self.max_points)
        self.subsampler.fit(self.X)

        self.X_t = self.mapper.transform(self.subsampler.transform(self.X))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: DPMClusteringModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.logging_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f'Visualizing decision boundaries at epoch {trainer.current_epoch}')
        X_t = self.X_t
        I = self.subsampler.transform(pl_module.val_r.argmax(dim=-1).cpu())

        mus = pl_module.cluster_gmm.mus.data.detach().cpu()
        covs = pl_module.cluster_gmm.covs.data.detach().cpu()
        k = pl_module.cluster_gmm.n_components

        # Remap Cluster Parameters
        if self.mapper:
            mus = self.mapper.transform(mus)
            covs = torch.stack([
                torch.eye(2) * self.mapper.transform(cov.diag().reshape(1, -1))
                for cov in covs
            ])

        colors = torch.tensor([self.cmap(i) for i in range(k)], dtype=torch.float)

        # Figure frame
        _min, _max = X_t.min(axis=0).values, X_t.max(axis=0).values
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 8))
        fig.tight_layout(rect=[0, -0.02, 1, 0.95])
        (ax_clusters, ax_boundaries) = axes

        # Plot both clusters and boundaries
        self.plot_clusters(ax_clusters, X_t, I, mus, covs, colors)
        cont = self.plot_decision_regions(ax_boundaries, X_t, I, colors, pl_module)

        # Crop the axes
        for ax in axes:
            ax.set_xlim([_min[0], _max[0]])
            ax.set_ylim([_min[1], _max[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        cbar = fig.colorbar(cont, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Max network response", rotation=270, labelpad=10, y=0.45)

        fig.suptitle(f'Epoch {trainer.current_epoch}')
        plt.show()

    def plot_clusters(self, ax, X_t, I, mus, covs, colors):
        for i, center in enumerate(mus):
            ell = self.draw_ellipse(ax, center, covs[i], color=colors[i])
            ax.add_artist(ell)

        ax.scatter(
            X_t[:, 0], X_t[:, 1],
            c=colors[I], alpha=0.5,
            s=self.marker_size, zorder=1
        )

        ax.plot(
            mus[:, 0], mus[:, 1],
            "ko", label="Net Centers",
            markersize=self.marker_size / 2,
            alpha=0.6, zorder=3
        )

        ax.set_title("Net Clusters and Covariances")

    def plot_decision_regions(self, ax, X_t, I, colors, pl_module: DPMClusteringModel):
        X_min, X_max = X_t.min(axis=0).values, X_t.max(axis=0).values
        arrays_for_meshgrid = [np.arange(X_min[d] - 0.1, X_max[d] + 0.1, 0.1) for d in range(X_t.shape[1])]
        xx, yy = np.meshgrid(*arrays_for_meshgrid)

        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        # horizontal stack vectors to create x1,x2 input for the model
        grid_t = np.hstack((r1, r2))
        grid = self.mapper.inverse_transform(torch.from_numpy(grid_t))
        yhat = pl_module.cluster_net(grid.float().to(pl_module.device))
        yhat_maxed = yhat.max(axis=1).values.cpu()

        cont = ax.contourf(xx, yy, yhat_maxed.reshape(xx.shape), alpha=0.5, cmap="jet")
        ax.scatter(
            X_t[:, 0],
            X_t[:, 1],
            c=colors[I],
            s=self.marker_size, zorder=1
        )
        ax.set_title("Decision Boundary")
        return cont

    def draw_ellipse(self, ax, mus, cov, color, **kwargs):
        covariance = cov  # np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariance)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(np.abs(v))
        ell = mpl.patches.Ellipse(mus, v[0], v[1], 180 + angle, color=color.numpy(), **kwargs)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        return ell
