from pathlib import Path
from typing import Optional

import numpy as np
import umap
import umap.plot
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import Tensor

from ml.layers.metrics import silhouette_score, davies_bouldin_score
from ml.models.dpm_clustering import DPMClusteringModel
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class ClusteringVisualizerCallback(Callback):
    mapper: umap.UMAP
    xs: Tensor
    sim: str

    def __init__(self, logging_interval: int = 3) -> None:
        super().__init__()
        self.logging_interval = logging_interval
        self.cmap = mpl.cm.get_cmap('tab10')

    def setup(self, trainer: Trainer, pl_module: DPMClusteringModel, stage: Optional[str] = None) -> None:
        logger.info('Pretraining UMAP mapping')
        self.sim = pl_module.hparams.sim
        self.xs = pl_module.dataset[torch.arange(len(pl_module.dataset), dtype=torch.long)]
        self.mapper = umap.UMAP(n_components=2).fit(self.xs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: DPMClusteringModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.logging_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f'Visualizing communities at epoch {trainer.current_epoch}')
        k = pl_module.k
        color_dict = {i: self.cmap(i / k) for i in range(k)}

        I = pl_module.val_r.argmax(dim=-1)
        centers = pl_module.cluster_gmm.mu.data.detach().cpu()

        ax = umap.plot.points(self.mapper, labels=I.numpy(), color_key=color_dict)

        centers = self.mapper.transform(centers.numpy())
        plt.scatter(
            centers[:, 0], centers[:, 1],
            c=list(color_dict.values()),
            marker='*', edgecolor='k',
            linewidths=mpl.rcParams['lines.linewidth'], s=mpl.rcParams['lines.markersize'] ** 3,
        )

        plt.title(f'Clustering at epoch {trainer.current_epoch}')
        plt.show()

