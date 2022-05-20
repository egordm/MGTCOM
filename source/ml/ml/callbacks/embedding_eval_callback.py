from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning import Trainer

from ml.algo.transforms import SubsampleTransform
from ml.callbacks.base.intermittent_callback import IntermittentCallback, IntermittentCallbackParams
from ml.evaluation import clustering_metrics
from ml.models.base.base_model import BaseModel
from ml.models.base.graph_datamodule import GraphDataModule
from ml.models.mgcom_e2e import MGCOME2EModel
from ml.utils import Metric, prefix_keys, dict_mapv
from ml.utils.training import ClusteringStage
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class EmbeddingEvalCallbackParams(IntermittentCallbackParams):
    metric: Metric = Metric.L2
    """Metric to use for embedding evaluation."""
    lp_max_pairs: int = 5000
    """Maximum number of pairs to use for link prediction."""
    met_max_points: int = 5000


class EmbeddingEvalCallback(IntermittentCallback[EmbeddingEvalCallbackParams]):
    def __init__(
            self,
            datamodule: GraphDataModule,
            hparams: EmbeddingEvalCallbackParams
    ) -> None:
        super().__init__(hparams)
        self.datamodule = datamodule

        self.val_subsample = SubsampleTransform(self.hparams.met_max_points)
        self.test_subsample = SubsampleTransform(self.hparams.met_max_points)

        self.val_labels = dict_mapv(datamodule.val_labels(), self.val_subsample.transform)
        self.test_labels = dict_mapv(datamodule.test_labels(), self.test_subsample.transform)

    def on_validation_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        if isinstance(pl_module, MGCOME2EModel) and pl_module.stage != ClusteringStage.Feature:
            return

        logger.info(f"Evaluating validation embeddings at epoch {trainer.current_epoch}")
        if pl_module.heterogeneous:
                Z = pl_module.val_outputs.extract_cat_kv('Z_dict', cache=True, device='cpu')
        else:
            Z = pl_module.val_outputs.extract_cat('Z', cache=True, device='cpu')

        Z = self.val_subsample.transform(Z)

        for label_name, labels in self.val_labels.items():
            pl_module.log_dict(prefix_keys(
                clustering_metrics(Z, labels, metric=self.hparams.metric),
                f'eval/val/ee/{label_name}/'
            ), on_epoch=True)

    def on_test_epoch_end_run(self, trainer: Trainer, pl_module: BaseModel) -> None:
        logger.info(f"Evaluating test embeddings")
        if pl_module.heterogeneous:
            Z = pl_module.test_outputs.extract_cat_kv('Z_dict', cache=True, device='cpu')
        else:
            Z = pl_module.test_outputs.extract_cat('Z', cache=True, device='cpu')

        Z = self.test_subsample.transform(Z)

        for label_name, labels in self.test_labels.items():
            pl_module.log_dict(prefix_keys(
                clustering_metrics(Z, labels, metric=self.hparams.metric),
                f'eval/test/ee/{label_name}/'
            ), on_epoch=True)
