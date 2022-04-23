from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from pytorch_lightning import Trainer, LightningModule, Callback
from torch import Tensor

from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.evaluation import silhouette_score, davies_bouldin_score
from ml.models.base.base import BaseModel
from ml.models.mgcom_comdet import MGCOMComDetDataModule
from ml.utils import HParams, Metric, prefix_keys
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClusteringEvalCallbackParams(HParams):
    ce_interval: int = 1
    """Interval between clustering evalutations."""
    metric: Metric = Metric.DOTP
    """Metric to use for embedding evaluation."""


class ClusteringEvalCallback(IntermittentCallback):
    def __init__(
            self,
            datamodule: MGCOMComDetDataModule,
            hparams: ClusteringEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or ClusteringEvalCallbackParams()
        super().__init__(self.hparams.ce_interval)
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if 'X' not in pl_module.val_outputs:
            return

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")
        X = pl_module.val_outputs.extract_cat('X', cache=True)
        r = pl_module.val_outputs.extract_cat('r', cache=True)
        z = r.argmax(dim=-1)

        trainer.logger.log_metrics(prefix_keys(
            self.clustering_metrics(X, z, metric=self.hparams.metric),
            'eval/val/clu/'
        ))

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if 'X' not in pl_module.val_outputs:
            return

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")
        X = pl_module.val_outputs.extract_cat('X', cache=True)
        r = pl_module.val_outputs.extract_cat('r', cache=True)
        z = r.argmax(dim=-1)

        trainer.logger.log_metrics(prefix_keys(
            self.clustering_metrics(X, z, metric=self.hparams.metric),
            'eval/test/clu/'
        ))

    @staticmethod
    def clustering_metrics(X: Tensor, z: Tensor, metric: Metric) -> Dict[str, float]:
        return {
            'silhouette_score': silhouette_score(X, z, metric=metric),
            'davies_bouldin_score': davies_bouldin_score(X, z, metric=metric),
        }
