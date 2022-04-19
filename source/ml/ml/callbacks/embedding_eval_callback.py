from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.trainer.states import RunningStage
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch_geometric.data import HeteroData

from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.data.graph_datamodule import GraphDataModule
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs, silhouette_score, davies_bouldin_score, link_prediction_measure
from ml.models.mgcom_feat import MGCOMFeatDataModule
from ml.utils import HParams, Metric
from ml.utils.labelling import NodeLabelling
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

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass  # NOOP: cause we dont collect embeddings on train

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        Z_dict = pl_module.val_Z_dict
        Z = torch.cat(list(Z_dict.values()), dim=0)

        val_metrics = self.collect_metrics(Z, self.val_labels, self.val_pairs)
        pl_module.log_dict({
            f'val_{metric}': value
            for metric, value in val_metrics.items()
        })

        train_metrics = self.collect_metrics(Z, self.train_labels, self.train_pairs)
        pl_module.log_dict({
            f'train_{metric}': value
            for metric, value in train_metrics.items()
        })

        pl_module.log_dict({
            'train_acc': train_metrics['link_prediction_accuracy'],
            'val_acc': val_metrics['link_prediction_accuracy']
        }, prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.state.stage != RunningStage.TESTING:
            return

        logger.info(f"Evaluating test embeddings")
        Z_dict = pl_module.test_Z_dict
        Z = torch.cat(list(Z_dict.values()), dim=0)

        metrics = self.collect_metrics(Z, self.test_labels, self.test_pairs)
        pl_module.log_dict({
            f'test_{metric}': value
            for metric, value in metrics.items()
        })

    def collect_metrics(
            self,
            Z: Tensor, labels: Dict[str, Tensor],
            lp_pairs: Tuple[Tensor, Tensor]
    ) -> Dict[str, float]:
        output = dict(
            link_prediction_accuracy=link_prediction_measure(Z, *lp_pairs, metric=self.hparams.metric),
        )

        for label_name, labels in labels.items():
            output.update({
                f'{metric} ({label_name})': value
                for metric, value in self.supervised_clustering_metrics(Z[:len(labels)], labels).items()
            })

        return output

    def supervised_clustering_metrics(self, Z: Tensor, labels: Tensor) -> Dict[str, float]:
        sc = silhouette_score(Z, labels, metric=self.hparams.metric)
        db = davies_bouldin_score(Z, labels, metric=self.hparams.metric)
        return dict(
            silhouette_score=sc,
            davies_bouldin_score=db
        )
