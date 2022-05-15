from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning import Trainer

from ml.algo.transforms import SubsampleTransform
from ml.callbacks.base.intermittent_callback import IntermittentCallback, IntermittentCallbackParams
from ml.evaluation import prediction_measure
from ml.models.base.base_model import BaseModel
from ml.models.base.graph_datamodule import GraphDataModule
from ml.models.mgcom_e2e import MGCOME2EModel
from ml.utils import Metric, prefix_keys, dict_mapv
from ml.utils.training import ClusteringStage
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class ClassificationEvalCallbackParams(IntermittentCallbackParams):
    enabled: bool = True
    """Whether to enable classification evaluation."""
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""
    cf_max_pairs: int = 5000
    """Maximum number of pairs to use for classification."""


class ClassificationEvalCallback(IntermittentCallback[ClassificationEvalCallbackParams]):
    def __init__(
            self,
            datamodule: GraphDataModule,
            hparams: ClassificationEvalCallbackParams
    ) -> None:
        super().__init__(hparams)
        self.datamodule = datamodule
        self.pairwise_dist_fn = self.hparams.metric.pairwise_dist_fn

        self.val_subsample = SubsampleTransform(self.hparams.cf_max_pairs)
        self.test_subsample = SubsampleTransform(self.hparams.cf_max_pairs)

        self.val_labels = dict_mapv(datamodule.val_labels(), self.val_subsample.transform)
        self.test_labels = dict_mapv(datamodule.test_labels(), self.test_subsample.transform)

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if not self.hparams.enabled or len(self.val_labels) == 0:
            return

        if isinstance(pl_module, MGCOME2EModel) and pl_module.stage != ClusteringStage.Feature:
            return

        logger.info(f"Evaluating validation classification at epoch {trainer.current_epoch}")
        if pl_module.heterogeneous:
            Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True, device='cpu')
        else:
            Z = pl_module.val_outputs.extract_cat('Z', cache=True, device='cpu')

        Z = self.val_subsample.transform(Z)

        for label_name, labels in self.val_labels.items():
            acc, metrics = prediction_measure(Z, labels, max_iter=1000)
            pl_module.log_dict(prefix_keys(metrics, f'eval/val/cf/{label_name}/'), on_epoch=True)

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if not self.hparams.enabled or len(self.test_labels) == 0:
            return

        logger.info(f"Evaluating test embeddings")
        if pl_module.heterogeneous:
            Z = pl_module.test_outputs.extract_cat_kv('Z_dict', cache=True)
        else:
            Z = pl_module.test_outputs.extract_cat('Z', cache=True)

        Z = self.test_subsample.transform(Z)

        for label_name, labels in self.test_labels.items():
            acc, metrics = prediction_measure(Z, labels, max_iter=1000)
            pl_module.log_dict(prefix_keys(metrics, f'eval/test/cf/{label_name}/'), on_epoch=True)
