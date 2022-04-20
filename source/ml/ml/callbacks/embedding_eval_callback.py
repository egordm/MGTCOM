from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

from ml.algo.transforms import SubsampleTransform
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
            data_module: GraphDataModule,
            hparams: EmbeddingEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or EmbeddingEvalCallbackParams()
        super().__init__()
        self.data_module = data_module
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.train_pairs = data_module.train_prediction_pairs()
        self.val_pairs = data_module.val_prediction_pairs()
        self.test_pairs = data_module.test_prediction_pairs()

        self.train_labels = {
            label_name: torch.cat(list(label_dict.values()), dim=0)
            for label_name, label_dict in data_module.train_inferred_labels().items()
        }
        self.val_labels = {
            label_name: torch.cat(list(label_dict.values()), dim=0)
            for label_name, label_dict in data_module.val_inferred_labels().items()
        }
        self.test_labels = {
            label_name: torch.cat(list(label_dict.values()), dim=0)
            for label_name, label_dict in data_module.test_inferred_labels().items()
        }

        self.transform_subsample_train = SubsampleTransform(self.hparams.met_max_points)
        self.transform_subsample_val = SubsampleTransform(self.hparams.met_max_points)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass  # NOOP: cause we dont collect embeddings on train

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        if trainer.current_epoch % self.hparams.ee_interval != 0 and trainer.current_epoch != trainer.max_epochs:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        Z_dict = pl_module.val_Z_dict
        Z = torch.cat(list(Z_dict.values()), dim=0)

        val_metrics = self.collect_metrics(Z, self.val_labels, self.val_pairs, self.transform_subsample_val)
        pl_module.log_dict({
            f'val/eval/{metric}': value
            for metric, value in val_metrics.items()
        })

        train_metrics = self.collect_metrics(Z, self.train_labels, self.train_pairs, self.transform_subsample_train)
        pl_module.log_dict({
            f'train/eval/{metric}': value
            for metric, value in train_metrics.items()
        })

        pl_module.log_dict({
            'train/eval/acc': train_metrics['link_prediction_accuracy'],
            'val/eval/acc': val_metrics['link_prediction_accuracy']
        }, prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        logger.info(f"Evaluating test embeddings")
        Z_dict = pl_module.test_Z_dict
        Z = torch.cat(list(Z_dict.values()), dim=0)

        metrics = self.collect_metrics(Z, self.test_labels, self.test_pairs)
        pl_module.log_dict({
            f'test/eval/{metric}': value
            for metric, value in metrics.items()
        })

    def collect_metrics(
            self,
            Z: Tensor, labels: Dict[str, Tensor],
            lp_pairs: Tuple[Tensor, Tensor],
            subsampler: SubsampleTransform = None
    ) -> Dict[str, float]:
        output = dict(
            link_prediction_accuracy=link_prediction_measure(Z, *lp_pairs, metric=self.hparams.metric),
        )

        for label_name, labels in labels.items():
            if subsampler is not None:
                Z_sub = subsampler.transform(Z[:len(labels)])
                labels = subsampler.transform(labels)
            else:
                Z_sub = Z

            output.update({
                f'{metric} ({label_name})': value
                for metric, value in self.supervised_clustering_metrics(Z_sub, labels).items()
            })

        return output

    def supervised_clustering_metrics(self, Z: Tensor, labels: Tensor) -> Dict[str, float]:
        sc = silhouette_score(Z, labels, metric=self.hparams.metric)
        db = davies_bouldin_score(Z, labels, metric=self.hparams.metric)
        return dict(
            silhouette_score=sc,
            davies_bouldin_score=db
        )
