from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from pytorch_lightning import Trainer, LightningModule
from torch import Tensor

from ml.callbacks.base.intermittent_callback import IntermittentCallback
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import silhouette_score, davies_bouldin_score, newman_girvan_modularity
from ml.evaluation.metrics.community import conductance
from ml.models.base.base_model import BaseModel
from ml.models.mgcom_comdet import MGCOMComDetDataModule
from ml.models.mgcom_e2e import MGCOME2EDataModule
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
            datamodule: Union[MGCOMComDetDataModule, MGCOME2EDataModule],
            hparams: ClusteringEvalCallbackParams = None
    ) -> None:
        self.hparams = hparams or ClusteringEvalCallbackParams()
        super().__init__(self.hparams.ce_interval)
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        if isinstance(datamodule, MGCOME2EDataModule) or datamodule.graph_dataset is not None:
            data = datamodule.val_data if isinstance(datamodule, MGCOME2EDataModule) else datamodule.graph_dataset
            hdata = to_homogeneous(
                data,
                node_attrs=[], edge_attrs=[],
                add_node_type=False, add_edge_type=False
            )
            self.val_edge_index = hdata.edge_index
        else:
            self.val_edge_index = None

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

        if self.val_edge_index is not None:
            metrics = self.community_metrics(z, self.val_edge_index)
            trainer.logger.log_metrics(prefix_keys(
                metrics,
                'eval/val/clu/'
            ))

            pl_module.log('modularity', metrics['modularity'], logger=False, prog_bar=True)
            pl_module.log('conductance', metrics['conductance'], logger=False, prog_bar=True)

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

        if self.val_edge_index is not None:
            trainer.logger.log_metrics(prefix_keys(
                self.community_metrics(z, self.val_edge_index),
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
            'conductance': conductance(edge_index, z),
        }
