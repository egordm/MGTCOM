from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from ml.algo.dpm import DPMMParams
from ml.algo.transforms import DimensionReductionMode, SubsampleTransform, DimensionReductionTransform
from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.models.dpmmsc import DPMMSubClusteringModel, Stage
from ml.utils import HParams, Metric
from ml.utils.plot import create_colormap, plot_scatter, draw_ellipses, MARKER_SIZE
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClusteringVisualizerCallbackParams(HParams):
    dim_reduction_mode: DimensionReductionMode = DimensionReductionMode.PCA
    """Dimension reduction mode for embedding visualization."""
    cv_max_points: int = 10000
    """Maximum number of points to visualize."""
    cv_interval: int = 1
    """Interval between clustering visualization."""
    metric: Metric = Metric.L2
    """Metric to use for embedding visualization."""


class ClusteringVisualizerCallback(IntermittentCallback):
    def __init__(self, hparams: ClusteringVisualizerCallbackParams = None) -> None:
        self.hparams = hparams or ClusteringVisualizerCallbackParams()
        super().__init__(self.hparams.cv_interval)

        if self.hparams.dim_reduction_mode == DimensionReductionMode.TSNE:
            logger.warning(f'Using PCA for visualization because TSNE does not support transfrom function.')
            self.hparams.dim_reduction_mode = DimensionReductionMode.PCA

        self.transform_subsample = SubsampleTransform(self.hparams.cv_max_points)
        self.transform_df = DimensionReductionTransform(
            n_components=2, mode=self.hparams.dim_reduction_mode, metric=self.hparams.metric
        )

    def _transform_params(self, params: DPMMParams) -> DPMMParams:
        mus = self.transform_df.transform(params.mus)
        sigmas = torch.stack([
            torch.eye(2) * self.transform_df.transform(cov.diag().reshape(1, -1))
            for cov in params.covs
        ])
        return DPMMParams(params.pis, mus, sigmas)

    def on_run(self, trainer: Trainer, pl_module: DPMMSubClusteringModel) -> None:
        if pl_module.stage == Stage.GatherSamples:
            return

        wandb_logger: WandbLogger = trainer.logger

        logger.info(f"Visualizing clustering at epoch {trainer.current_epoch}")
        visualize_subclusters = pl_module.is_subclustering

        # Collect sample data
        k = pl_module.k
        X = pl_module.val_outputs['X']
        X = self.transform_subsample.transform(X)

        r = pl_module.val_outputs['r']
        r = self.transform_subsample.transform(r)
        z = r.argmax(dim=-1)

        if visualize_subclusters:
            ri = pl_module.val_outputs['ri']
            ri = self.transform_subsample.transform(ri)
            zi = ri.argmax(dim=-1)
        else:
            ri, zi = None, None

        # Transform sample data
        if not self.transform_df.is_fitted or pl_module.samplespace_changed:
            logger.info(f'Transforming embeddings using {self.hparams.dim_reduction_mode}...')
            self.transform_df.fit(X)

        X = self.transform_df.transform(X)

        # Collect cluster params
        cluster_params = pl_module.cluster_params
        cluster_params = self._transform_params(cluster_params)
        if visualize_subclusters:
            subcluster_params = pl_module.subcluster_params
            subcluster_params = self._transform_params(subcluster_params)
        else:
            subcluster_params = None

        fig = self.visualize(
            X, z, zi,
            k, cluster_params, subcluster_params,
            pl_module,
            title=f'Epoch {trainer.current_epoch}'
        )
        wandb_logger.log_metrics({
            f'visualization/Clustering Visualization': wandb.Image(fig)
        })
        if wandb.run.offline:
            plt.show()
        else:
            plt.close(fig)

        self.visualize_distributions(
            trainer, k,
            pl_module.clusters.pis, pl_module.subcluster_params.pis if visualize_subclusters else None,
        )

    def visualize(
            self, X: Tensor, z: Tensor, zi: Tensor,
            k, cluster_params: DPMMParams, subcluster_params: DPMMParams,
            pl_module: DPMMSubClusteringModel, title: str = ''
    ):
        colors = create_colormap(k)

        # Figure frame
        _min, _max = X.min(dim=0).values, X.max(dim=0).values
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 6))
        fig.tight_layout(rect=[0, -0.02, 1, 0.95])
        (ax_clusters, ax_boundaries) = axes

        # Plot both clusters and boundaries
        self.plot_clusters(
            ax_clusters, X, z, zi,
            cluster_params, subcluster_params,
            colors
        )
        cont = self.plot_decision_regions(ax_boundaries, X, z, colors, pl_module)

        # Crop the axes
        for ax in axes:
            ax.set_xlim([_min[0], _max[0]])
            ax.set_ylim([_min[1], _max[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        cbar = fig.colorbar(cont, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Max network response", rotation=270, labelpad=10, y=0.45)

        fig.suptitle(title)
        return fig

    def visualize_distributions(self, trainer: Trainer, k: int, pis: Tensor, pis_sub: Tensor = None):
        fig, (axc, axsc) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(12, 6))
        ind = np.arange(k)
        axc.bar(ind, pis, label="Cluster Weights", align="center", alpha=0.3)
        axc.set_xlabel("Clusters (k={})".format(k))
        axc.set_ylabel("Normalized weights")
        axc.set_title(f"Epoch {trainer.current_epoch}: Clusters weights")

        if pis_sub is not None:
            pi_sub_1 = pis_sub[0::2]
            pi_sub_2 = pis_sub[1::2]
            axsc.bar(ind, pi_sub_1, align="center", label="Sub Cluster 1")
            axsc.bar(ind, pi_sub_2, align="center", bottom=pi_sub_1, label="Sub Cluster 2")
            axsc.set_xlabel("Clusters (k={})".format(k))
            axsc.set_ylabel("Normalized weights")
            axsc.set_title(f"Epoch {trainer.current_epoch}: Clusters weights")

        trainer.logger.log_metrics({
            f'visualization/Cluster Distribution': wandb.Image(fig)
        })
        if wandb.run.offline:
            plt.show()
        else:
            plt.close(fig)

    def plot_clusters(
            self, ax, X: Tensor, z: Tensor, zi: Tensor,
            cluster_params: DPMMParams, subcluster_params: DPMMParams,
            colors
    ):
        # Draw points
        plot_scatter(
            ax, X[:, 0], X[:, 1],
            facecolors=colors[z], alpha=0.6,
            markers=['+', 'x'], marker_idx=zi,
            linewidth=2, s=MARKER_SIZE * 4, zorder=3
        )

        # Draw clusters
        plot_scatter(
            ax, cluster_params.mus[:, 0], cluster_params.mus[:, 1],
            marker="o", facecolor="k", edgecolors=colors,
            label="Net Centers",
            s=MARKER_SIZE * 8, linewidth=2,
            alpha=0.6, zorder=4
        )
        draw_ellipses(ax, cluster_params.mus, cluster_params.covs, colors, alpha=0.2, zorder=1)

        # Draw subclusters
        if subcluster_params is not None:
            indices = (torch.arange(len(subcluster_params.mus)) / 2).floor().long()

            draw_ellipses(ax, subcluster_params.mus, subcluster_params.covs, colors[indices], alpha=0.2, zorder=2)
            plot_scatter(
                ax, subcluster_params.mus[:, 0], subcluster_params.mus[:, 1],
                edgecolor="k", facecolors=colors[indices],
                label="Net Centers",
                markers=['<', '>'], marker_idx=(torch.arange(len(subcluster_params.mus)) % 2),
                linewidth=2, s=MARKER_SIZE * 5, zorder=5
            )

        ax.set_title("Net Clusters and Covariances")

    def plot_decision_regions(self, ax, X, z, colors, pl_module: DPMMSubClusteringModel):
        X_min, X_max = X.min(axis=0).values, X.max(axis=0).values
        arrays_for_meshgrid = [np.arange(X_min[d] - 0.1, X_max[d] + 0.1, 0.1) for d in range(X.shape[1])]
        xx, yy = np.meshgrid(*arrays_for_meshgrid)

        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        # horizontal stack vectors to create x1,x2 input for the model
        grid_t = np.hstack((r1, r2))
        grid = self.transform_df.inverse_transform(torch.from_numpy(grid_t))
        yhat = pl_module.estimate_assignment(grid.float().to(pl_module.device))
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
