from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Type

from pytorch_lightning import LightningDataModule, Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from simple_parsing import Serializable
from transformers.models.longformer.convert_longformer_original_pytorch_lightning_to_pytorch import LightningModel

from datasets import GraphDataset
from ml.callbacks.progress_bar import CustomProgressBar
from ml.callbacks.save_config_callback import SaveConfigCallback
from ml.utils import DataLoaderParams, OptimizerParams, TrainerParams
from shared import parse_args, get_logger, RESULTS_PATH


@dataclass
class BaseExecutorArgs(Serializable):
    loader_params: DataLoaderParams = DataLoaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    trainer_params: TrainerParams = TrainerParams()


class BaseExecutor:
    args: BaseExecutorArgs
    datamodule: LightningDataModule
    model: LightningModel
    callbacks: List[Callback]

    TASK_NAME = 'base'
    EXECUTOR_NAME: str
    RUN_NAME: str

    def __init__(self) -> None:
        super().__init__()
        self.EXECUTOR_NAME = self.__class__.__name__
        self.logger = get_logger(self.EXECUTOR_NAME)

    @abstractmethod
    def params_cls(self) -> Type[BaseExecutorArgs]:
        raise NotImplementedError

    def cli(self):
        self.args = parse_args(self.params_cls())[0]
        self.datamodule = self.datamodule()
        self.RUN_NAME = self.run_name()

        root_dir = RESULTS_PATH / self.TASK_NAME / self.EXECUTOR_NAME / self.RUN_NAME
        root_dir.mkdir(exist_ok=True, parents=True)

        self.callbacks = [
            CustomProgressBar(),
            LearningRateMonitor(logging_interval='step'),
            SaveConfigCallback(self.args),
            *self.callbacks()
        ]

        self.model = self.model()

        trainer = Trainer(
            **self.args.trainer_params.to_dict(),
            default_root_dir=str(root_dir),
            callbacks=self.callbacks,
            # num_sanity_val_steps=0,
        )

        self.logger.info(f'Training {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        trainer.fit(self.model, self.datamodule)

        self.logger.info(f'Testing {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        trainer.test(self.model, self.datamodule)

        self.logger.info(f'Predicting {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        trainer.predict(self.model, self.datamodule)

    @abstractmethod
    def datamodule(self) -> LightningDataModule:
        raise NotImplementedError

    @abstractmethod
    def model(self) -> LightningModel:
        raise NotImplementedError

    @abstractmethod
    def callbacks(self) -> List[Callback]:
        raise NotImplementedError

    def run_name(self):
        return 'base'
