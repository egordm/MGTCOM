from dataclasses import dataclass
from pathlib import Path
from typing import Union

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
from ml.models.mgcom_comdet import MGCOMComDetModel, Stage
from ml.models.mgcom_e2e import MGCOME2EModel, Stage as StageE2E
from ml.utils import HParams, Metric
from ml.utils.plot import create_colormap, plot_scatter, draw_ellipses, MARKER_SIZE, plot_decision_regions
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
    metric: Metric = Metric.DOTP
    """Metric to use for embedding visualization."""


class ClusteringVisualizerCallback(IntermittentCallback):
    sample_space_version: int = -1

    def __init__(self, hparams: ClusteringVisualizerCallbackParams = None) -> None:
        self.hparams = hparams or ClusteringVisualizerCallbackParams()
        super().__init__(self.hparams.cv_interval)

        if self.hparams.dim_reduction_mode == DimensionReductionMode.TSNE:
            logger.warning(f'Using PCA for visualization because TSNE does not support transfrom function.')
            self.hparams.dim_reduction_mode = DimensionReductionMode.PCA

        self.subsample = SubsampleTransform(self.hparams.cv_max_points)
        self.remap = DimensionReductionTransform(
            n_components=2, mode=self.hparams.dim_reduction_mode, metric=self.hparams.metric
        )

    def _transform_params(self, params: DPMMParams) -> DPMMParams:
        mus = self.remap.transform(params.mus)
        sigmas = torch.stack([
            torch.eye(2) * self.remap.transform(cov.diag().reshape(1, -1))
            for cov in params.covs
        ])
        return DPMMParams(params.pis, mus, sigmas)

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: Union[MGCOMComDetModel, MGCOME2EModel]) -> None:
        if isinstance(pl_module, MGCOME2EModel):
            if pl_module.stage == StageE2E.Feature or pl_module.val_outputs is None:
                return

            pl_module = pl_module.clustering_model

        if pl_module.stage == Stage.GatherSamples:
            return

        logger.info(f"Visualizing clustering at epoch {trainer.current_epoch}")
        visualize_subclusters = pl_module.hparams.subcluster

        # Collect sample data
        k = pl_module.dpmm_model.k
        X = self.subsample.transform(pl_module.val_outputs.extract_cat('X', cache=True))

        r = self.subsample.transform(pl_module.val_outputs.extract_cat('r', cache=True))
        z = r.argmax(dim=-1)

        if visualize_subclusters:
            ri = self.subsample.transform(pl_module.val_outputs.extract_cat('ri', cache=True))
            zi = ri.argmax(dim=-1)
        else:
            ri, zi = None, None

        # Transform sample data
        if not self.remap.is_fitted or pl_module.sample_space_version != self.sample_space_version:
            logger.info(f'Transforming embeddings using {self.hparams.dim_reduction_mode}...')
            self.remap.fit(X)
            self.sample_space_version = pl_module.sample_space_version

        X = self.remap.transform(X)

        # Collect cluster params
        cluster_params = self._transform_params(pl_module.dpmm_model.cluster_params)
        if visualize_subclusters:
            subcluster_params = self._transform_params(pl_module.dpmm_model.subcluster_params)
        else:
            subcluster_params = None

        fig = self.visualize_clusters(
            X, z, zi,
            k, cluster_params, subcluster_params,
            pl_module,
            title=f'Epoch {trainer.current_epoch}'
        )
        # noinspection PyTypeChecker
        trainer.logger.log_metrics({
            f'viz/cluster_centers': wandb.Image(fig)
        })
        if wandb.run.offline:
            plt.show()
        else:
            plt.close(fig)

        if not wandb.run.offline:
            fig = self.visualize_distributions(
                trainer, k,
                cluster_params.pis, subcluster_params.pis if visualize_subclusters else None,
            )
            # noinspection PyTypeChecker
            trainer.logger.log_metrics({
                f'viz/cluster_distribution': wandb.Image(fig)
            })
            plt.close(fig)

    def visualize_clusters(
            self, X: Tensor, z: Tensor, zi: Tensor,
            k, cluster_params: DPMMParams, subcluster_params: DPMMParams,
            pl_module: MGCOMComDetModel, title: str = ''
    ):
        colors = create_colormap(k)

        # Figure frame
        _min, _max = X.min(dim=0).values, X.max(dim=0).values
        fig, axes = plt.subplots(nrows=1, ncols=3 if subcluster_params is not None else 2, sharey=True, figsize=(10, 6))
        fig.tight_layout(rect=[0, -0.02, 1, 0.95])
        (*ax_clusters, ax_boundaries) = axes

        # Plot both clusters and subclusters
        self.plot_clusters(ax_clusters[-1], X, z, cluster_params, colors)
        if subcluster_params is not None:
            self.plot_subclusters(ax_clusters[0], X, z, zi, subcluster_params, colors)

        # Plot boundaries
        cont = plot_decision_regions(
            ax_boundaries, X, z, colors,
            lambda x: pl_module.dpmm_model.estimate_assignment(self.remap.inverse_transform(x))
        )
        cbar = fig.colorbar(cont, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Max network response", rotation=270, labelpad=10, y=0.45)

        # Crop the axes
        for ax in axes:
            ax.set_xlim([float(_min[0]), float(_max[0])])
            ax.set_ylim([float(_min[1]), float(_max[1])])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        fig.suptitle(title)
        return fig

    def plot_clusters(
            self, ax, X: Tensor, z: Tensor,
            cluster_params: DPMMParams, colors
    ):
        # Draw points
        plot_scatter(
            ax, X[:, 0], X[:, 1],
            facecolors=colors[z], alpha=0.6,
            # markers=['+', 'x'], marker_idx=zi,
            linewidth=2, s=MARKER_SIZE * 4, zorder=3
        )

        # Draw clusters
        plot_scatter(
            ax, cluster_params.mus[:, 0], cluster_params.mus[:, 1],
            marker="o", facecolor="k", edgecolors=colors,
            label="Net Centers",
            s=MARKER_SIZE * 8, linewidth=2,
            alpha=0.6, zorder=3
        )
        draw_ellipses(ax, cluster_params.mus, cluster_params.covs, colors, alpha=0.2, zorder=1)

        ax.set_title("Net Clusters and Covariances")

    def plot_subclusters(self, ax, X: Tensor, z: Tensor, zi: Tensor, subcluster_params: DPMMParams, colors):
        # Draw points
        plot_scatter(
            ax, X[:, 0], X[:, 1],
            facecolors=colors[z], alpha=0.6,
            markers=['+', 'x'], marker_idx=zi,
            linewidth=2, s=MARKER_SIZE * 4, zorder=3
        )

        indices = (torch.arange(len(subcluster_params.mus)) / 2).floor().long()

        # Draw subclusters
        plot_scatter(
            ax, subcluster_params.mus[:, 0], subcluster_params.mus[:, 1],
            edgecolor="k", facecolors=colors[indices],
            label="Net Centers",
            markers=['<', '>'], marker_idx=(torch.arange(len(subcluster_params.mus)) % 2),
            linewidth=2, s=MARKER_SIZE * 5, zorder=3
        )
        draw_ellipses(ax, subcluster_params.mus, subcluster_params.covs, colors[indices], alpha=0.2, zorder=1)

        ax.set_title("Net SubClusters and Covariances")

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

        return fig
