from functools import partial
from pathlib import Path

import torch
from pytorch_lightning.loops import FitLoop, PredictionEpochLoop
from pytorch_lightning.trainer.progress import Progress
from torch import Tensor

from ml.algo.dpmm.base import EMCallback, BaseMixture
from ml.algo.transforms import ToHeteroMappingTransform
from ml.models.mgcom_e2e import MGCOME2EDataModule, MGCOME2EModel
from ml.utils.outputs import OutputExtractor
from ml.utils.training import ClusteringStage
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class E2EFitLoop(FitLoop, EMCallback):
    def __init__(
        self,
        min_epochs: int = 0,
        max_epochs: int = 1000,
        n_pretrain_epochs: int = 20,
        n_feat_epochs: int = 20,
        n_cluster_epochs: int = 100,
    ) -> None:
        super().__init__(min_epochs, max_epochs)
        self.epoch_feat_progress = Progress()
        self.epoch_cluster_progress = Progress()

        self.n_pretrain_epochs = n_pretrain_epochs
        self.n_feat_epochs = n_feat_epochs
        self.n_cluster_epochs = n_cluster_epochs

        self.cluster_embed_loop = PredictionEpochLoop()
        self.sample_space_version = 0
        self.pretraining = True
        self.X = None

    def run(self):
        if self.skip:
            return self.on_skip()

        self.reset()

        self.on_run_start()

        while not self.done:
            try:
                logger.info(f'Starting feature stage: Epoch {self.trainer.current_epoch}')
                self.run_feat()
                logger.info(f'Starting clustering stage: Epoch {self.trainer.current_epoch}')
                self.run_cluster()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False

        output = self.on_run_end()
        return output

    def run_feat(self) -> None:
        self.model.stage = ClusteringStage.Feature
        for i in range(self.n_pretrain_epochs if self.pretraining else self.n_feat_epochs):
            if self.done:
                break

            self.epoch_feat_progress.increment_ready()
            self.epoch_feat_progress.increment_started()
            self.on_advance_start()
            self.advance()
            self.epoch_feat_progress.increment_processed()
            self.epoch_feat_progress.increment_completed()
            self.on_advance_end()
            self.model.sample_space_version += 1

    def run_cluster(self) -> None:
        self.model.stage = ClusteringStage.GatherSamples
        dataloader = self.datamodule.cluster_dataloader()
        dataloader = self.trainer.strategy.process_dataloader(dataloader)
        self._data_fetcher.setup(
            dataloader, batch_to_device=partial(self.trainer._call_strategy_hook, "batch_to_device", dataloader_idx=0)
        )
        with torch.no_grad():
            X = []
            for batch_idx, batch in enumerate(self._data_fetcher):
                X.append(self.model.forward_homogenous(batch))

        self.X = torch.cat(X, dim=0).cpu()

        self.model.stage = ClusteringStage.Clustering
        self.model.cluster_model.fit(
            self.X,
            n_init=1,
            max_iter=self.n_cluster_epochs,
            incremental=self.model.cluster_model.is_fitted,
            callbacks=[self],
        )
        self.model.r_prev = self.model.cluster_model.estimate_log_resp(self.X).exp().to(self.model.device)
        self.X = None

    def on_before_step(self, model: 'BaseMixture') -> None:
        self.on_advance_start()
        self.epoch_feat_progress.increment_ready()
        self.epoch_feat_progress.increment_started()

    def on_after_step(self, _model: BaseMixture, lower_bound: Tensor) -> None:
        z, zi = self.model.cluster_model.predict_full(self.X)
        self.model.val_outputs = OutputExtractor([{'X': self.X, 'z': z, 'zi': zi}])

        self.epoch_feat_progress.increment_processed()
        self.epoch_feat_progress.increment_completed()
        self.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.increment_completed()
        self.on_advance_end()
        self.trainer._call_callback_hooks("on_validation_epoch_end")
        self.trainer._call_callback_hooks("on_validation_end")
        self.model.pretraining = False
        self.pretraining = False

    @property
    def model(self) -> MGCOME2EModel:
        return self.trainer.model

    @property
    def datamodule(self) -> MGCOME2EDataModule:
        return self.trainer.datamodule
