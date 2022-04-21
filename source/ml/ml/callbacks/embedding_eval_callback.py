from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

from ml.algo.transforms import SubsampleTransform
from ml.models.base.embedding import BaseModel
from ml.models.base.graph_datamodule import GraphDataModule
from ml.evaluation import silhouette_score, davies_bouldin_score, link_prediction_measure
from ml.utils import HParams, Metric
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class EmbeddingEvalCallbackParams(HParams):
    ee_interval: int = 1
    """Interval between embedding evalutations."""
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""
    lp_max_pairs: int = 5000
    """Maximum number of pairs to use for link prediction."""
    met_max_points: int = 5000


class EmbeddingEvalCallback(Callback):
    def __init__(
            self,
            datamodule: GraphDataModule,
            hparams: EmbeddingEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or EmbeddingEvalCallbackParams()
        super().__init__()
        self.datamodule = datamodule
        # self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.val_subsample = SubsampleTransform(self.hparams.met_max_points)
        self.test_subsample = SubsampleTransform(self.hparams.met_max_points)

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

        if trainer.current_epoch % self.hparams.ee_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        if pl_module.heterogeneous:
            Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.val_outputs.extract_cat('Z', cache=True)

        Z = self.val_subsample.transform(Z)

        for label_name, labels in self.val_labels.items():
            pl_module.log_dict({
                f'eval/val/com/{label_name}/{name}': value
                for name, value in self.supervised_clustering_metrics(Z, labels).items()
            })

    def on_test_epoch_end(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        logger.info(f"Evaluating test embeddings")
        if pl_module.heterogeneous:
            Z = pl_module.test_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.test_outputs.extract_cat('Z', cache=True)

        Z = self.test_subsample.transform(Z)

        for label_name, labels in self.test_labels.items():
            pl_module.log_dict({
                f'eval/val/com/{label_name}/{name}': value
                for name, value in self.supervised_clustering_metrics(Z, labels).items()
            })

    def supervised_clustering_metrics(self, Z: Tensor, labels: Tensor) -> Dict[str, float]:
        sc = silhouette_score(Z, labels, metric=self.hparams.metric)
        db = davies_bouldin_score(Z, labels, metric=self.hparams.metric)
        return dict(
            silhouette_score=sc,
            davies_bouldin_score=db
        )
