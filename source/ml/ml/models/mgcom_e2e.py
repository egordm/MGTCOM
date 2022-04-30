from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Union, List

from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EPOCH_OUTPUT
from torch_geometric.typing import Metadata, NodeType

from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.loaders.nodes_loader import HeteroNodesLoader
from ml.models.base.feature_model import HeteroFeatureModel
from ml.models.mgcom_combi import MGCOMCombiModel, MGCOMCombiModelParams, MGCOMCombiDataModule
from ml.models.mgcom_comdet import MGCOMComDetModel, MGCOMComDetModelParams, Stage as StageDPMM
from ml.models.node2vec import UnsupervisedLoss
from ml.utils import HParams, OptimizerParams


class Stage(Enum):
    Feature = 0
    Clustering = 1


@dataclass
class MGCOME2EModelParams(HParams):
    combi_params: MGCOMCombiModelParams = MGCOMCombiModelParams(use_cluster=True, loss=UnsupervisedLoss.HINGE)
    cluster_params: MGCOMComDetModelParams = MGCOMComDetModelParams()

    pretrain_epochs: int = 20
    feat_epochs: int = 10
    cluster_epochs: int = 20


class MGCOME2EModel(HeteroFeatureModel):
    hparams: Union[MGCOME2EModelParams, OptimizerParams]

    def __init__(
            self,
            metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
            hparams: MGCOME2EModelParams,
            optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())

        self.clustering_model = MGCOMComDetModel(
            MGCOMCombiModel.compute_repr_dim(hparams.combi_params),
            hparams=hparams.cluster_params,
            optimizer_params=optimizer_params,
        )

        self.combi_model = MGCOMCombiModel(
            metadata, num_nodes_dict,
            hparams=hparams.combi_params,
            optimizer_params=optimizer_params,
            cluster_module=self.clustering_model,
        )

        self.stage = Stage.Feature
        self.epoch_counter = 0

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.combi_model.trainer = self.trainer
        self.clustering_model.trainer = self.trainer

    @property
    def repr_dim(self):
        return self.combi_model.repr_dim

    def forward(self, *args, **kwargs) -> Any:
        return self.combi_model.forward(*args, **kwargs)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.combi_model.on_train_start()
        self.clustering_model.on_train_start()

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        if self.stage == Stage.Feature:
            return self.combi_model.training_step(batch, batch_idx)
        elif self.stage == Stage.Clustering:
            _, node_perm_dict = batch
            Z_dict = self.combi_model(batch)
            Z = ToHeteroMappingTransform.inverse_transform_values(
                Z_dict, node_perm_dict, shape=[self.repr_dim], device=self.device
            ).detach()
            return self.clustering_model.training_step(Z, batch_idx)

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)
        self.combi_model._current_fx_name = self._current_fx_name
        self.clustering_model._current_fx_name = self._current_fx_name

        if self.stage == Stage.Feature:
            self.combi_model.training_epoch_end(outputs)
        elif self.stage == Stage.Clustering:
            self.clustering_model.training_epoch_end(outputs)

        if self.current_epoch == self.hparams.pretrain_epochs:
            self.combi_model.pretraining = False
            self.set_stage(Stage.Clustering)

        if self.clustering_model.is_done:
            self.set_stage(Stage.Feature)

        self.epoch_counter += 1
        if self.stage == Stage.Feature \
                and self.epoch_counter >= self.hparams.feat_epochs \
                and self.current_epoch >= self.hparams.pretrain_epochs:
            self.set_stage(Stage.Clustering) # TODO: Update prior here somewhere?
            self.clustering_model.stage = StageDPMM.GatherSamples
            # Z = self.train_outputs.extract_cat('Z', cache=True)
            # self.clustering_model.dpmm_model.reinitialize(Z, incremental=True)

        if self.stage == Stage.Clustering \
                and self.epoch_counter >= self.hparams.cluster_epochs \
                and self.current_epoch >= self.hparams.pretrain_epochs:
            self.set_stage(Stage.Feature)

        self.log_dict({
            'stage_feat': self.stage == Stage.Feature,
            'stage_clus': self.stage == Stage.Clustering,
        })

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        if self.stage == Stage.Feature:
            return super().validation_step(batch, batch_idx)
        elif self.stage == Stage.Clustering:
            _, node_perm_dict = batch
            Z_dict = self.combi_model(batch)
            Z = ToHeteroMappingTransform.inverse_transform_values(
                Z_dict, node_perm_dict, shape=[self.repr_dim], device=self.device
            ).detach()

            return {
                **self.clustering_model.validation_step(Z, batch_idx),
                'Z': Z,
                'Z_dict': Z_dict,
            }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().validation_epoch_end(outputs)

        if self.stage == Stage.Clustering:
            self.clustering_model.validation_epoch_end(outputs)

    def set_stage(self, stage: Stage):
        prev_stage = self.stage
        self.stage = stage

        if self.stage == Stage.Feature:
            self.automatic_optimization = True
        elif self.stage == Stage.Clustering:
            self.automatic_optimization = False
            self.clustering_model.dpmm_model.burnin_monitor.reset()

        if self.trainer is not None:
            self.trainer.datamodule.set_stage(stage)

            if prev_stage != stage:
                self.trainer.reset_train_dataloader(self)
                self.epoch_counter = 0

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        return {
            **super().get_progress_bar_dict(),
            'stage': 'feat' if self.stage == Stage.Feature else 'clus',
            'k': self.clustering_model.k,
        }


class MGCOME2EDataModule(MGCOMCombiDataModule):
    stage: Stage = Stage.Feature

    def set_stage(self, stage: Stage):
        self.stage = stage

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.stage == Stage.Feature:
            return super().train_dataloader()
        elif self.stage == Stage.Clustering:
            return HeteroNodesLoader(
                self.train_data.num_nodes_dict,
                transform_nodes_fn=self.eval_sampler(self.train_data),
                shuffle=False,
                **self.loader_params.to_dict()
            )
