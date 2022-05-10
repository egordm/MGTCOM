import shutil
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, Optional, TypeVar, Generic

import wandb
from pytorch_lightning import LightningDataModule, Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from simple_parsing import Serializable

from ml.callbacks.classification_eval_callback import ClassificationEvalCallback, ClassificationEvalCallbackParams
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallbackParams
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallbackParams
from ml.callbacks.embedding_eval_callback import EmbeddingEvalCallbackParams, EmbeddingEvalCallback
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallbackParams, EmbeddingVisualizerCallback
from ml.callbacks.lp_eval_callback import LPEvalCallbackParams, LPEvalCallback
from ml.callbacks.progress_bar import CustomProgressBar
from ml.callbacks.save_config_callback import SaveConfigCallback, MyDumper
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallbackParams, SaveGraphCallback
from ml.callbacks.save_modelsummary_callback import SaveModelSummaryCallback
from ml.models.base.graph_datamodule import GraphDataModule
from ml.utils import DataLoaderParams, OptimizerParams, TrainerParams, Metric, recursively_override_attr
from shared import parse_args, get_logger, RESULTS_PATH


@dataclass
class CallbackArgs(Serializable):
    clustering_visualizer: ClusteringVisualizerCallbackParams = ClusteringVisualizerCallbackParams()
    clustering_eval: ClusteringEvalCallbackParams = ClusteringEvalCallbackParams()
    embedding_visualizer: EmbeddingVisualizerCallbackParams = EmbeddingVisualizerCallbackParams()
    embedding_eval: EmbeddingEvalCallbackParams = EmbeddingEvalCallbackParams()
    save_graph: SaveGraphCallbackParams = SaveGraphCallbackParams()
    lp_eval: LPEvalCallbackParams = LPEvalCallbackParams()
    classification_eval: ClassificationEvalCallbackParams = ClassificationEvalCallbackParams()


@dataclass
class BaseExecutorArgs(Serializable):
    project: str = "ThesisExperiments"
    experiment: Optional[str] = None
    run_name: Optional[str] = None
    show_config: bool = False
    dry_run: bool = False
    debug: bool = False
    offline: bool = False
    metric: Optional[Metric] = None

    loader_params: DataLoaderParams = DataLoaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    trainer_params: TrainerParams = TrainerParams()
    callback_params: CallbackArgs = CallbackArgs()


T = TypeVar("T", bound=LightningModule)


class BaseExecutor(Generic[T]):
    args: BaseExecutorArgs
    datamodule: LightningDataModule
    model: T
    callbacks: List[Callback]
    wandb_logger: WandbLogger
    trainer: Trainer
    checkpoint_callback: ModelCheckpoint
    root_dir: Path

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
        if self.args.metric is not None:
            self.logger.info(f'Using metric {self.args.metric} globally')
            recursively_override_attr(self.args, 'metric', self.args.metric)

        self.datamodule = self._datamodule()
        self.RUN_NAME = self.run_name()

        if self.args.show_config:
            print('=' * 80)
            print(self.args.dumps_yaml(Dumper=MyDumper))
            return

        if self.args.debug:
            self.args.trainer_params.cpu = True
            self.args.loader_params.num_workers = 0

        self.root_dir = RESULTS_PATH / self.TASK_NAME / self.EXECUTOR_NAME / self.RUN_NAME
        self.root_dir.mkdir(exist_ok=True, parents=True)

        self.callbacks = [
            CustomProgressBar(),
            LearningRateMonitor(logging_interval='step'),
            SaveConfigCallback(self.args),
            SaveModelSummaryCallback(),
            *self._callbacks()
        ]
        self.model = self.model_args(self.model_cls)
        self.wandb_logger = self._logger()
        self.checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor=self._metric_monitor(),
            mode='max',
        )
        self.trainer = self._trainer(callbacks=[
            *self.callbacks,
            self.checkpoint_callback
        ])

        self._fit()

    def _fit(self):
        self.logger.info(f'Training {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        if not self.args.dry_run:
            self.trainer.fit(self.model, self.datamodule)

        if not self.args.dry_run and self.checkpoint_callback.best_model_path:
            self.logger.info(f'Saving best model {Path(self.checkpoint_callback.best_model_path).name}')
            shutil.copyfile(
                src=self.checkpoint_callback.best_model_path,
                dst=Path(wandb.run.dir) / 'best_model.ckpt'
            )

        if not self.args.dry_run and self.checkpoint_callback.best_model_path:
            self.logger.info(f'Loading best model: {Path(self.checkpoint_callback.best_model_path).name}')
            model_args, model_kwargs = self.model_args(lambda *args, **kwargs: (args, kwargs))
            self.model = self.model.load_from_checkpoint(self.checkpoint_callback.best_model_path, *model_args,
                **model_kwargs)

        self.logger.info(f'Testing {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        if not self.args.dry_run:
            self.trainer.test(self.model, self.datamodule)

        self.logger.info(f'Predicting {self.TASK_NAME}/{self.EXECUTOR_NAME}/{self.RUN_NAME}')
        if not self.args.dry_run:
            self.trainer.predict(self.model, self.datamodule)

    @abstractmethod
    def _datamodule(self) -> LightningDataModule:
        raise NotImplementedError

    @abstractmethod
    def model_args(self, cls: Type[T]) -> T:
        raise NotImplementedError

    @abstractmethod
    def _callbacks(self) -> List[Callback]:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_cls(self) -> Type[T]:
        raise NotImplementedError

    def before_training(self, trainer: Trainer):
        pass

    def run_name(self):
        return 'base'

    def tags(self):
        tags = [self.TASK_NAME]

        if hasattr(self.args, 'dataset'):
            tags.append(str(self.args.dataset))

        return tags

    def _trainer(self, **kwargs) -> Trainer:
        trainer_args = self.args.trainer_params.to_dict()
        trainer_args.pop('cpu', None)
        trainer = Trainer(
            **trainer_args,
            default_root_dir=str(self.root_dir),
            logger=self.wandb_logger,
            gpus=1 if not self.args.trainer_params.cpu else None,
            auto_lr_find=True,
            enable_model_summary=False,
            **kwargs,
        )
        return trainer

    def _logger(self, **kwargs) -> WandbLogger:
        run_config = self.args.to_dict()
        run_config.pop('wandb_project_name', None)
        run_config.pop('run_name', None)
        run_config.pop('show_config', None)
        run_config.pop('offline', None)

        model_name = self.model.__class__.__name__
        wandb_config = {
            'args': run_config,
            'model': model_name,
        }
        if hasattr(self.args, 'dataset'):
            wandb_config['dataset'] = self.args.dataset

        wandb_args = {}
        if self.args.run_name is not None:
            wandb_args['name'] = self.args.run_name

        return WandbLogger(
            project=self.args.project,
            group=self.args.experiment,
            save_dir=str(self.root_dir),
            config=wandb_config,
            tags=self.tags(),
            job_type=self.TASK_NAME,
            offline=self.args.offline,
            **wandb_args,
            **kwargs
        )

    def _embedding_task_callbacks(self) -> List[Callback]:
        if not isinstance(self.datamodule, GraphDataModule):
            raise ValueError('Embedding task callbacks only work with GraphDataModule')

        return [
            EmbeddingVisualizerCallback(
                self.datamodule,
                hparams=self.args.callback_params.embedding_visualizer
            ),
            EmbeddingEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.embedding_eval
            ),
            LPEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.lp_eval,
            ),
            ClassificationEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.classification_eval,
            ),
            SaveGraphCallback(
                self.datamodule.dataset,
                hparams=self.args.callback_params.save_graph
            ),
            SaveEmbeddingsCallback(),
        ]

    def _metric_monitor(self) -> str:
        return 'val/lp/acc'
