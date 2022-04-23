from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from torch import Tensor

from ml.algo.transforms import DimensionReductionMode, DimensionReductionTransform, SubsampleTransform
from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.models.base.base_model import BaseModel
from ml.utils import HParams, Metric
from ml.utils.labelling import NodeLabelling
from ml.utils.plot import plot_scatter, create_colormap, MARKER_SIZE
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class EmbeddingVisualizerCallbackParams(HParams):
    dim_reduction_mode: DimensionReductionMode = DimensionReductionMode.UMAP
    # dim_reduction_mode: DimensionReductionMode = DimensionReductionMode.TSNE
    """Dimension reduction mode for embedding visualization."""
    ev_max_points: int = 1000
    """Maximum number of points to visualize."""
    ev_interval: int = 20
    """Interval between embedding visualization."""
    metric: Metric = Metric.L2
    """Metric to use for embedding visualization."""


class EmbeddingVisualizerCallback(IntermittentCallback):
    def __init__(self, val_node_labels: Dict[str, NodeLabelling],
                 hparams: EmbeddingVisualizerCallbackParams = None) -> None:
        self.hparams = hparams or EmbeddingVisualizerCallbackParams()
        super().__init__(interval=self.hparams.ev_interval)

        self.val_subsample = SubsampleTransform(self.hparams.ev_max_points)
        self.mapper = DimensionReductionTransform(
            n_components=2, mode=self.hparams.dim_reduction_mode, metric=self.hparams.metric
        )

        self.val_labels = {
            label_name: self.val_subsample.transform(torch.cat(list(labels_dict.values()), dim=0))
            for label_name, labels_dict in val_node_labels.items()
        }

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if trainer.current_epoch == 0:
            return

        logger.info(f"Visualizing embeddings at epoch {trainer.current_epoch}")
        Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True)
        Z = self.val_subsample.transform(Z)

        logger.info(f'Transforming embeddings using {self.hparams.dim_reduction_mode}...')
        self.mapper.fit(Z)
        Z = self.mapper.transform(Z)

        for label_name, labels in self.val_labels.items():
            fig = self.visualize_embeddings(
                Z, labels,
                f'Epoch {trainer.current_epoch} - Embedding Visualization ({label_name})'
            )

            # noinspection PyTypeChecker
            trainer.logger.log_metrics({
                f'visualization/Embedding Visualization ({label_name})': wandb.Image(fig)
            })

            if wandb.run.offline:
                plt.show()

    def visualize_embeddings(self, Z: Tensor, labels: Tensor, title):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
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
        return fig
