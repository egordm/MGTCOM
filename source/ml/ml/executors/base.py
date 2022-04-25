from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Type, Optional

from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule, Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from simple_parsing import Serializable
from transformers.models.longformer.convert_longformer_original_pytorch_lightning_to_pytorch import LightningModel

from ml.callbacks.classification_eval_callback import ClassificationEvalCallback, ClassificationEvalCallbackParams
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback, ClusteringEvalCallbackParams
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
from ml.utils import DataLoaderParams, OptimizerParams, TrainerParams
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
    wandb_project_name: str = "ThesisDebug"
    run_name: Optional[str] = None
    show_config: bool = False
    debug: bool = False
    offline: bool = False

    loader_params: DataLoaderParams = DataLoaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    trainer_params: TrainerParams = TrainerParams()
    callback_params: CallbackArgs = CallbackArgs()


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

        if self.args.show_config:
            print('=' * 80)
            print(self.args.dumps_yaml(Dumper=MyDumper))
            return

        if self.args.debug:
            self.args.trainer_params.cpu = True
            self.args.loader_params.num_workers = 0

        root_dir = RESULTS_PATH / self.TASK_NAME / self.EXECUTOR_NAME / self.RUN_NAME
        root_dir.mkdir(exist_ok=True, parents=True)

        self.callbacks = [
            CustomProgressBar(),
            LearningRateMonitor(logging_interval='step'),
            SaveConfigCallback(self.args),
            SaveModelSummaryCallback(),
            *self.callbacks()
        ]

        self.model = self.model()

        # Initialize wandb logger
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

        wandb_logger = WandbLogger(
            project=self.args.wandb_project_name,
            save_dir=str(root_dir),
            config=wandb_config,
            tags=self.tags(),
            job_type=self.TASK_NAME,
            offline=self.args.offline,
            **wandb_args
        )

        trainer_args = self.args.trainer_params.to_dict()
        trainer_args.pop('cpu', None)
        trainer = Trainer(
            **trainer_args,
            default_root_dir=str(root_dir),
            callbacks=self.callbacks,
            logger=wandb_logger,
            gpus=1 if not self.args.trainer_params.cpu else None,
            auto_lr_find=True,
            enable_model_summary=False,
        )
        self.before_training(trainer)

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

    def before_training(self, trainer: Trainer):
        pass

    def run_name(self):
        return 'base'

    def tags(self):
        tags = [self.TASK_NAME]

        if hasattr(self.args, 'dataset'):
            tags.append(str(self.args.dataset))

        return tags

    def _embedding_task_callbacks(self) -> List[Callback]:
        if not isinstance(self.datamodule, GraphDataModule):
            raise ValueError('Embedding task callbacks only work with GraphDataModule')

        return [
            EmbeddingVisualizerCallback(
                val_node_labels=self.datamodule.val_inferred_labels(),
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
                self.datamodule.data,
                node_labels=self.datamodule.inferred_labels(),
                hparams=self.args.callback_params.save_graph
            ),
            SaveEmbeddingsCallback(),
        ]
