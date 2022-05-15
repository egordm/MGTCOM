from collections import ChainMap
from enum import Enum

from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

from ml.models.base.base_model import BaseModel
from ml.utils.outputs import OutputExtractor


class EvaluationMode(Enum):
    TEST = 'test'
    VALIDATION = 'validation'


def mock_evaluation_epoch(
    trainer: Trainer, model: BaseModel,
    outputs: OutputExtractor,
    mode=EvaluationMode.TEST
) -> None:
    trainer.state.stage = RunningStage.TESTING if mode == EvaluationMode.TEST else RunningStage.VALIDATING
    trainer.state.fn = TrainerFn.TESTING if mode == EvaluationMode.TEST else TrainerFn.VALIDATING
    prefix = 'test' if mode == EvaluationMode.TEST else 'validation'

    trainer._call_callback_hooks(f"on_{prefix}_start")
    trainer._call_lightning_module_hook(f"on_{prefix}_start")
    trainer._call_strategy_hook(f"on_{prefix}_start")
    trainer._call_lightning_module_hook(f"on_{prefix}_model_eval")

    trainer._logger_connector.on_epoch_start()
    trainer._call_callback_hooks(f"on_{prefix}_epoch_start")
    trainer._call_lightning_module_hook(f"on_{prefix}_epoch_start")

    if mode == EvaluationMode.TEST:
        model.test_outputs = outputs
    else:
        model.val_outputs = outputs

    trainer._logger_connector.epoch_end_reached()

    trainer._call_callback_hooks(f'on_{prefix}_epoch_end')
    trainer._call_lightning_module_hook(f'on_{prefix}_epoch_end')
    trainer._logger_connector.on_epoch_end()

    logged_outputs, _logged_outputs = [], []  # free memory
    epoch_end_logged_outputs = trainer._logger_connector.update_eval_epoch_metrics()
    all_logged_outputs = dict(ChainMap(*logged_outputs))  # list[dict] -> dict
    all_logged_outputs.update(epoch_end_logged_outputs)
    for dl_outputs in logged_outputs:
        dl_outputs.update(epoch_end_logged_outputs)

    # log metrics
    trainer._logger_connector.log_eval_end_metrics(all_logged_outputs)

    trainer._call_callback_hooks(f"on_{prefix}_end")
    trainer._call_lightning_module_hook(f"on_{prefix}_end")
    trainer._call_strategy_hook(f"on_{prefix}_end")
