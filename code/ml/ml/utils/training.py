from enum import IntEnum

from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage, TrainerFn


def override_trainer_state(model: LightningModule, state: RunningStage, method: str = None):
    model.trainer.state.stage = state
    if state == RunningStage.TRAINING:
        model.trainer.state.fn = TrainerFn.FITTING
    elif state == RunningStage.VALIDATING:
        model.trainer.state.fn = TrainerFn.VALIDATING
    elif state == RunningStage.TESTING:
        model.trainer.state.fn = TrainerFn.TESTING

    model._current_fx_name = method or 'on_test_epoch_end'


class ClusteringStage(IntEnum):
    GatherSamples = 0
    Clustering = 1
    Feature = 2
