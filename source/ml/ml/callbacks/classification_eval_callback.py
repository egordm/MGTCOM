from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage

from ml.algo.transforms import SubsampleTransform
from ml.models.base.graph_datamodule import GraphDataModule
from ml.evaluation import link_prediction_measure, prediction_measure
from ml.models.base.embedding import BaseModel
from ml.utils import HParams, Metric, merge_dicts
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClassificationEvalCallbackParams(HParams):
    metric: Metric = Metric.DOTP
    """Metric to use for embedding evaluation."""
    cl_max_pairs: int = 5000
    """Maximum number of pairs to use for classification."""


class ClassificationEvalCallback(Callback):
    def __init__(
            self,
            datamodule: GraphDataModule,
            hparams: ClassificationEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or ClassificationEvalCallbackParams()
        super().__init__()
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.val_subsample = SubsampleTransform(self.hparams.cl_max_pairs)
        self.test_subsample = SubsampleTransform(self.hparams.cl_max_pairs)

        self.val_labels = {
            label_name: self.val_subsample.transform(torch.cat(list(labels_dict.values()), dim=0))
            for label_name, labels_dict in datamodule.val_inferred_labels().items()
        }
        self.test_labels = {
            label_name: self.test_subsample.transform(torch.cat(list(labels_dict.values()), dim=0))
            for label_name, labels_dict in datamodule.test_inferred_labels().items()
        }

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if len(self.val_labels) == 0:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        if pl_module.heterogeneous:
            Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.val_outputs.extract_cat('Z', cache=True)

        Z = self.val_subsample.transform(Z)

        for label_name, labels in self.val_labels.items():
            acc, metrics = prediction_measure(Z, labels)

            pl_module.log_dict({
                f'eval/val/cl/{label_name}/{name}': value
                for name, value in metrics.items()
            })

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        if len(self.val_labels) == 0:
            return

        logger.info(f"Evaluating test embeddings")
        if pl_module.heterogeneous:
            Z = pl_module.test_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.test_outputs.extract_cat('Z', cache=True)

        Z = self.test_subsample.transform(Z)

        for label_name, labels in self.test_labels.items():
            acc, metrics = prediction_measure(Z, labels)

            pl_module.log_dict({
                f'eval/test/cl/{label_name}/{name}': value
                for name, value in metrics.items()
            })
