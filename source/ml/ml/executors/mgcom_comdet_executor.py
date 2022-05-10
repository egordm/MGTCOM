from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, List, Any

import torch
from pytorch_lightning import LightningDataModule, Callback, Trainer
from pytorch_lightning.loops import FitLoop, EvaluationLoop
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from torch import Tensor
from torch.utils.data import Dataset

from datasets import GraphDataset
from datasets.utils.conversion import igraph_from_hetero
from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.algo.dpmm.base import EMCallback, BaseMixture
from ml.algo.dpmm.statistics import InitMode
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.data import PretrainedEmbeddingsDataset, SyntheticGMMDataset
from ml.executors.base import BaseExecutorArgs, BaseExecutor, T
from ml.models.mgcom_comdet import MGCOMComDetDataModuleParams, MGCOMComDetDataModule, MGCOMComDetModel, \
    MGCOMComDetModelParams
from ml.utils import dataset_choices, DataLoaderParams, TrainerParams
from ml.utils.loops.TrainlessFitLoop import TrainlessFitLoop
from ml.utils.outputs import OutputExtractor
from ml.utils.training import override_trainer_state
from shared import get_logger

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'community_detection'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args(BaseExecutorArgs):
    dataset: str = dataset_choices()
    """Graph Dataset to use for visualization and evaluation."""
    pretrained_path: Optional[str] = None
    hparams: MGCOMComDetModelParams = MGCOMComDetModelParams()
    data_params: MGCOMComDetDataModuleParams = MGCOMComDetDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams(batch_size=200)
    trainer_params: TrainerParams = TrainerParams(max_epochs=200)


class MGCOMComDetExecutor(BaseExecutor[MGCOMComDetModel]):
    args: Args
    datamodule: MGCOMComDetDataModule
    dataset: Dataset

    TASK_NAME = 'community_detection'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        graph_dataset: Optional[GraphDataset] = DATASET_REGISTRY[self.args.dataset]()

        if self.args.pretrained_path:
            logger.info(f'Using pretrained embeddings from {self.args.pretrained_path}')
            dataset = PretrainedEmbeddingsDataset.from_pretrained(self.args.pretrained_path, graph_dataset.name)
        else:
            logger.info('No pretrained embeddings provided, using synthetic dataset')
            dataset = SyntheticGMMDataset()
            graph_dataset = None

        self.dataset = dataset

        return MGCOMComDetDataModule(
            dataset=dataset,
            graph_dataset=graph_dataset,
            hparams=self.args.data_params,
            loader_params=self.args.loader_params,
        )

    def model_args(self, cls):
        return cls(
            hparams=self.args.hparams
        )

    def _fit(self):
        self.logger.info('Training model')
        args, model, trainer = self.args, self.model, self.trainer
        # self.model.trainer = self.trainer
        # self.trainer.strategy.model = self.model

        loader = self.datamodule.predict_dataloader()
        X = torch.cat([batch for batch in loader], dim=0)
        # z_init = torch.cat(list(self.datamodule.graph_dataset.data.louvain_dict.values()), dim=0)
        z_init = None

        class MyLoop(TrainlessFitLoop, EMCallback):
            def do_advance_loop(self):
                self._restarting = False
                self.on_advance_start()
                model.cluster_model.fit(
                    X, callbacks=[self], z_init=z_init,
                    max_iter=args.trainer_params.max_epochs,
                    n_init=args.hparams.n_restart,
                )
                self.on_advance_end()

            def on_after_step(self, _model: BaseMixture, lower_bound: Tensor) -> None:
                self.advance()
                z, zi = model.cluster_model.predict_full(X)
                model.val_outputs = OutputExtractor([{'X': X, 'z': z, 'zi': zi}])

                self.on_advance_end()
                self.on_advance_start()

        trainer.fit_loop = MyLoop()
        trainer.fit(model, self.datamodule)

        override_trainer_state(self.model, RunningStage.TESTING, 'on_test_epoch_end')
        self.model.test_outputs = self.model.val_outputs
        trainer._call_callback_hooks('on_test_epoch_end')
        trainer._call_callback_hooks('on_predict_epoch_end', self.model.test_outputs.outputs)

    @property
    def model_cls(self) -> Type[MGCOMComDetModel]:
        return MGCOMComDetModel

    def _callbacks(self) -> List[Callback]:
        ret = [
            ClusteringVisualizerCallback(
                hparams=self.args.callback_params.clustering_visualizer
            ),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
            SaveEmbeddingsCallback(),
        ]

        if self.datamodule.graph_dataset is not None:
            ret.append(
                SaveGraphCallback(
                    self.datamodule.graph_dataset,
                    hparams=self.args.callback_params.save_graph,
                    clustering=False,
                )
            )

        return ret

    def run_name(self):
        return self.args.dataset


if __name__ == '__main__':
    MGCOMComDetExecutor().cli()
