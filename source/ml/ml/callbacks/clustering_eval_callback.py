from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage

from ml.evaluation import silhouette_score, davies_bouldin_score
from ml.models.dpmmsc import Stage, DPMMSubClusteringModel
from ml.models.mgcom_comdet import MGCOMComDetDataModule
from ml.utils import HParams, Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClusteringEvalCallbackParams(HParams):
    ce_interval: int = 1
    """Interval between clustering evalutations."""
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""


class ClusteringEvalCallback(Callback):
    def __init__(
            self,
            datamodule: MGCOMComDetDataModule,
            hparams: ClusteringEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or ClusteringEvalCallbackParams()
        super().__init__()
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass  # NOOP: cause we dont collect embeddings on train

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: DPMMSubClusteringModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.hparams.ce_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        if pl_module.stage == Stage.GatherSamples:
            return

        wandb_logger: WandbLogger = trainer.logger
        eval_subclusters = pl_module.is_subclustering

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")

        X = pl_module.val_outputs['X']
        r = pl_module.val_outputs['r']
        z = r.argmax(dim=-1)
        k = pl_module.k

        sc = silhouette_score(X, z, metric=self.hparams.metric)
        db = davies_bouldin_score(X, z, metric=self.hparams.metric)

        wandb_logger.log_metrics({
            'eval/train/silhouette_score': sc,
            'eval/train/davies_bouldin_score': db,
        })

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        if pl_module.stage == Stage.GatherSamples:
            return

        wandb_logger: WandbLogger = trainer.logger
        eval_subclusters = pl_module.is_subclustering

        logger.info(f"Evaluating test clustering")

        X = pl_module.val_outputs['X']
        r = pl_module.val_outputs['r']
        z = r.argmax(dim=-1)
        k = pl_module.k

        sc = silhouette_score(X, z, metric=self.hparams.metric)
        db = davies_bouldin_score(X, z, metric=self.hparams.metric)

        wandb_logger.log_metrics({
            'eval/test/silhouette_score': sc,
            'eval/test/davies_bouldin_score': db,
        })
