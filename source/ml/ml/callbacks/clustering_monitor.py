from pathlib import Path
from typing import Optional

import torch
import umap.plot
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

from ml.layers.metrics import silhouette_score, davies_bouldin_score
from ml.models.dpm_clustering import DPMClusteringModel
from ml.utils import Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class ClusteringMonitor(Callback):
    xs: Tensor
    metric: Metric

    def __init__(self, logging_interval: int = 1) -> None:
        super().__init__()
        self.logging_interval = logging_interval

    def setup(self, trainer: Trainer, pl_module: DPMClusteringModel, stage: Optional[str] = None) -> None:
        self.metric = pl_module.hparams.metric
        self.xs = pl_module.dataset[torch.arange(len(pl_module.dataset), dtype=torch.long)]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: DPMClusteringModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.logging_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f'Computing cluster scores {trainer.current_epoch}')
        k = pl_module.k
        I = pl_module.val_r.argmax(dim=-1)

        # modularity = newman_girvan_modularity(self.data, I, self.k)
        sc = silhouette_score(self.xs, I, metric=self.metric)
        db = davies_bouldin_score(self.xs, I, metric=self.metric)

        pl_module.log_dict({
            # "epoch_m": modularity,
            "epoch_sc": sc,
            "epoch_dbs": db,
        }, prog_bar=True, logger=True)
