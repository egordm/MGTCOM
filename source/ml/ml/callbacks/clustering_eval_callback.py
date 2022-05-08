from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from pytorch_lightning import Trainer, LightningModule
from torch import Tensor
from torch_geometric.data import HeteroData, Data

from datasets.transforms.to_homogeneous import to_homogeneous
from ml.callbacks.base.intermittent_callback import IntermittentCallback, IntermittentCallbackParams
from ml.evaluation import clustering_metrics, \
    community_gt_metrics
from ml.evaluation.metrics.community import community_metrics
from ml.models.base.base_model import BaseModel
from ml.models.base.graph_datamodule import GraphDataModule
from ml.models.mgcom_comdet import MGCOMComDetDataModule
from ml.utils import Metric, prefix_keys
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClusteringEvalCallbackParams(IntermittentCallbackParams):
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""


def extract_edge_index(data) -> Optional[Tensor]:
    if isinstance(data, HeteroData):
        hdata = to_homogeneous(
            data,
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        return hdata.edge_index
    elif isinstance(data, Data):
        return data.edge_index
    else:
        return None


class ClusteringEvalCallback(IntermittentCallback[ClusteringEvalCallbackParams]):
    def __init__(
            self,
            datamodule: Union[MGCOMComDetDataModule, GraphDataModule],
            hparams: ClusteringEvalCallbackParams
    ) -> None:
        super().__init__(hparams)
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        if isinstance(datamodule, GraphDataModule):
            self.val_edge_index = extract_edge_index(datamodule.val_data)
            self.test_edge_index = extract_edge_index(datamodule.test_data)
        elif hasattr(datamodule, 'graph_dataset') and datamodule.graph_dataset is not None:
            self.val_edge_index = extract_edge_index(datamodule.graph_dataset.data)
            self.test_edge_index = self.val_edge_index
        else:
            self.val_edge_index = None
            self.test_edge_index = None

        if isinstance(datamodule, GraphDataModule):
            self.val_labels = datamodule.val_labels()
            self.test_labels = datamodule.test_labels()
        else:
            # TODO: what else
            self.val_labels = None
            self.test_labels = None

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if 'X' not in pl_module.val_outputs or 'r' not in pl_module.val_outputs:
            return

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")
        X = pl_module.val_outputs.extract_cat('X', cache=True, device='cpu')
        r = pl_module.val_outputs.extract_cat('r', cache=True, device='cpu')
        z = r.argmax(dim=-1)

        pl_module.log_dict(prefix_keys(
            clustering_metrics(X, z, metric=self.hparams.metric), 'eval/val/clu/'
        ), on_epoch=True)

        if self.val_edge_index is not None:
            metrics = community_metrics(z, self.val_edge_index)
            pl_module.log_dict(prefix_keys(metrics, 'eval/val/clu/'), on_epoch=True)
            pl_module.log('modularity', metrics['modularity'], logger=False, prog_bar=True)

        if self.val_labels is not None:
            for label_name, labels in self.val_labels.items():
                metrics = community_gt_metrics(z, labels)
                pl_module.log_dict(prefix_keys(metrics, f'eval/val/clu/{label_name}/'), on_epoch=True)

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if 'X' not in pl_module.val_outputs:
            return

        logger.info(f"Evaluating validation clustering at epoch {trainer.current_epoch}")
        X = pl_module.val_outputs.extract_cat('X', cache=True, device='cpu')
        r = pl_module.val_outputs.extract_cat('r', cache=True, device='cpu')
        z = r.argmax(dim=-1)

        pl_module.log_dict(prefix_keys(
            clustering_metrics(X, z, metric=self.hparams.metric), 'eval/test/clu/'
        ), on_epoch=True)

        if self.test_edge_index is not None:
            pl_module.log_dict(prefix_keys(
                community_metrics(z, self.test_edge_index), 'eval/test/clu/'
            ), on_epoch=True)

        if self.test_labels is not None:
            for label_name, labels in self.test_labels.items():
                metrics = community_gt_metrics(z, labels)
                pl_module.log_dict(prefix_keys(metrics, f'eval/val/clu/{label_name}/'), on_epoch=True)
