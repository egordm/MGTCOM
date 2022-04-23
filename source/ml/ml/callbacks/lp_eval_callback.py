from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage

from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.models.base.graph_datamodule import GraphDataModule
from ml.evaluation import link_prediction_measure
from ml.models.base.embedding import BaseModel, BaseEmbeddingModel
from ml.utils import HParams, Metric, prefix_keys
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class LPEvalCallbackParams(HParams):
    metric: Metric = Metric.DOTP
    """Metric to use for embedding evaluation."""
    lp_max_pairs: int = 5000
    """Maximum number of pairs to use for link prediction."""


class LPEvalCallback(IntermittentCallback):
    def __init__(
            self,
            datamodule: GraphDataModule,
            hparams: LPEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or LPEvalCallbackParams()
        super().__init__(1)
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.val_pairs = datamodule.val_prediction_pairs()
        self.test_pairs = datamodule.test_prediction_pairs()

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseEmbeddingModel) -> None:
        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        if pl_module.heterogeneous:
            Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.val_outputs.extract_cat('Z', cache=True)

        acc, metrics = link_prediction_measure(Z, *self.val_pairs, metric=self.hparams.metric)

        pl_module.log_dict({
            f'val/lp/acc': acc
        }, prog_bar=True, logger=False)

        trainer.logger.log_metrics(prefix_keys(metrics, 'eval/val/lp/'))

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: BaseEmbeddingModel) -> None:
        logger.info(f"Evaluating test embeddings")
        if pl_module.heterogeneous:
            Z = pl_module.test_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.test_outputs.extract_cat('Z', cache=True)

        _, metrics = link_prediction_measure(Z, *self.test_pairs, metric=self.hparams.metric)

        trainer.logger.log_metrics(prefix_keys(metrics, 'eval/test/lp/'))
