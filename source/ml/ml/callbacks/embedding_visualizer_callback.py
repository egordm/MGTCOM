from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, LightningModule
from torch import Tensor

from ml.algo.transforms import DimensionReductionMode, DimensionReductionTransform, SubsampleTransform
from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.utils import HParams, Metric
from ml.utils.plot import plot_scatter, create_colormap, MARKER_SIZE
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class EmbeddingVisualizerCallbackParams(HParams):
    dim_reduction_mode: DimensionReductionMode = DimensionReductionMode.TSNE
    """Dimension reduction mode for embedding visualization."""
    ev_max_points: int = 10000
    """Maximum number of points to visualize."""
    ev_interval: int = 3
    """Interval between embedding visualization."""
    metric: Metric = Metric.L2
    """Metric to use for embedding visualization."""


class EmbeddingVisualizerCallback(IntermittentCallback):
    def __init__(self, node_labels: Dict[str, Tensor], hparams: EmbeddingVisualizerCallbackParams = None) -> None:
        self.hparams = hparams or EmbeddingVisualizerCallbackParams()
        super().__init__(interval=self.hparams.ev_interval)
        self.node_labels = node_labels

        num_points = len(next(iter(self.node_labels.values())))
        self.transform_subsample = SubsampleTransform(self.hparams.ev_max_points)
        self.transform_subsample.fit(num_points)

        self.transform_df = DimensionReductionTransform(
            n_components=2, mode=self.hparams.dim_reduction_mode, metric=self.hparams.metric
        )

    def on_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger.info(f"Visualizing embeddings at epoch {trainer.current_epoch}")
        Z = pl_module.val_Z
        Z = self.transform_subsample.transform(Z)

        logger.info(f'Transforming embeddings using {self.hparams.dim_reduction_mode}...')
        self.transform_df.fit(Z)
        Z = self.transform_df.transform(Z)

        for label_name, labels in self.node_labels.items():
            labels = self.transform_subsample.transform(labels)
            self.visualize_embeddings(
                Z, labels,
                f'Epoch {trainer.current_epoch} - Embedding Visualization ({label_name})'
            )

    def visualize_embeddings(self, Z: Tensor, labels: Tensor, title) -> None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(10, 6))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        num_labels = int(labels.max() + 1)
        colors = create_colormap(num_labels)

        # Draw points
        plot_scatter(
            ax, Z[:, 0], Z[:, 1],
            facecolors=colors[labels], alpha=0.6,
            linewidth=2, s=MARKER_SIZE * 4, zorder=3
        )

        fig.suptitle(title)
        plt.show()
