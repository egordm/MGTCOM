from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from torch import Tensor

from datasets.utils.conversion import igraph_from_hetero
from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import silhouette_score, davies_bouldin_score, newman_girvan_modularity, \
    newman_girvan_modularity_hetero
from ml.models.base.base_model import BaseModel
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

        if datamodule.graph_dataset is not None:
            hdata = to_homogeneous(
                datamodule.graph_dataset.data,
                node_attrs=[], edge_attrs=[],
                add_node_type=False, add_edge_type=False
            )
            self.edge_index = hdata.edge_index
        else:
            self.edge_index = None

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if 'X' not in pl_module.val_outputs or 'r' not in pl_module.val_outputs:
            return

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")
        X = pl_module.val_outputs.extract_cat('X', cache=True)
        r = pl_module.val_outputs.extract_cat('r', cache=True)
        z = r.argmax(dim=-1)

        trainer.logger.log_metrics(prefix_keys(
            self.clustering_metrics(X, z, metric=self.hparams.metric),
            'eval/val/clu/'
        ))

        if self.edge_index is not None:
            metrics = self.community_metrics(z, self.edge_index)
            trainer.logger.log_metrics(prefix_keys(
                metrics,
                'eval/val/clu/'
            ))

            pl_module.log('modularity', metrics['modularity'], logger=False, prog_bar=True)

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

        if self.edge_index is not None:
            trainer.logger.log_metrics(prefix_keys(
                self.community_metrics(z, self.edge_index),
                'eval/val/clu/'
            ))

    @staticmethod
    def clustering_metrics(X: Tensor, z: Tensor, metric: Metric) -> Dict[str, float]:
        return {
            'silhouette_score': silhouette_score(X, z, metric=metric),
            'davies_bouldin_score': davies_bouldin_score(X, z, metric=metric),
        }

    @staticmethod
    def community_metrics(z: Tensor, edge_index: Tensor) -> Dict[str, float]:
        return {
            'modularity': newman_girvan_modularity(edge_index, z),
        }
