from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage

from ml.models.base.graph_datamodule import GraphDataModule
from ml.evaluation import link_prediction_measure
from ml.models.base.embedding import BaseModel
from ml.utils import HParams, Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class LPEvalCallbackParams(HParams):
    lp_interval: int = 1
    """Interval between embedding evalutations."""
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""
    lp_max_pairs: int = 5000
    """Maximum number of pairs to use for link prediction."""


class LPEvalCallback(Callback):
    def __init__(
            self,
            data_module: GraphDataModule,
            hparams: LPEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or LPEvalCallbackParams()
        super().__init__()
        self.data_module = data_module
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.train_pairs = data_module.train_prediction_pairs()
        self.val_pairs = data_module.val_prediction_pairs()
        self.test_pairs = data_module.test_prediction_pairs()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass  # NOOP: cause we dont collect embeddings on train

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.hparams.lp_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        if self.data_module.heterogenous:
            Z_dict = pl_module.val_outputs['Z_dict']
            Z = torch.cat(list(Z_dict.values()), dim=0)
        else:
            Z = pl_module.val_outputs['Z']

        acc = link_prediction_measure(Z, *self.val_pairs, metric=self.hparams.metric)

        pl_module.log_dict({
            f'eval/val/lp_acc': acc
        }, prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        logger.info(f"Evaluating test embeddings")
        if self.data_module.heterogenous:
            Z_dict = pl_module.val_outputs['Z_dict']
            Z = torch.cat(list(Z_dict.values()), dim=0)
        else:
            Z = pl_module.val_outputs['Z']

        acc = link_prediction_measure(Z, *self.val_pairs, metric=self.hparams.metric)

        pl_module.log_dict({
            f'eval/test/lp_acc': acc
        })