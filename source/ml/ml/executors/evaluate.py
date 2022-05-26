from collections import ChainMap
from dataclasses import dataclass
from typing import Type, List

import numpy as np
import torch
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

from datasets.utils.graph_dataset import DATASET_REGISTRY
from ml.callbacks.clustering_eval_callback import ClusteringEvalCallback
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.callbacks.embedding_eval_callback import EmbeddingEvalCallback
from ml.evaluation.utils.mock import mock_evaluation_epoch, EvaluationMode
from ml.executors.base import BaseExecutorArgs, BaseExecutor, T
from ml.models.base.base_model import BaseModel
from ml.models.base.feature_model import FeatureModel
from ml.models.base.graph_datamodule import GraphDataModule, GraphDataModuleParams
from ml.utils import dataset_choices, DataLoaderParams
from ml.utils.outputs import OutputExtractor
from shared import parse_args, EXPORTS_PATH, OUTPUTS_PATH, RESULTS_PATH


@dataclass
class Args(BaseExecutorArgs):
    model: str = None
    run_name: str = None
    dataset: str = dataset_choices()
    dataset_version: str = 'base'
    k: int = -1
    repr_dim: int = -1
    baseline: bool = True


class EvaluateExecutor(BaseExecutor):
    args: Args

    TASK_NAME = 'baseline'

    def params_cls(self) -> Type[BaseExecutorArgs]:
        return Args

    def _datamodule(self) -> LightningDataModule:
        dataset = DATASET_REGISTRY[self.args.dataset]()
        return GraphDataModule(dataset, hparams=GraphDataModuleParams(), loader_params=DataLoaderParams())

    def _callbacks(self) -> List[Callback]:
        return [
            *self._embedding_task_callbacks(),
            ClusteringVisualizerCallback(
                hparams=self.args.callback_params.clustering_visualizer
            ),
            ClusteringEvalCallback(
                self.datamodule,
                hparams=self.args.callback_params.clustering_eval
            ),
        ]

    def model_args(self, cls: Type[T]) -> T:
        return FeatureModel()

    @property
    def model_cls(self) -> Type[T]:
        return self.args.model

    def _fit(self):
        datasets_path = EXPORTS_PATH / self.args.dataset / self.args.dataset_version
        output_path = OUTPUTS_PATH / self.args.model / self.args.dataset / self.args.dataset_version / self.args.run_name

        assert datasets_path.exists(), f'Datasets path does not exist: {datasets_path}'
        assert output_path.exists(), f'Output path does not exist: {output_path}'

        # Load Data
        train_data = torch.load(datasets_path / 'train.pt')
        val_data = torch.load(datasets_path / 'val.pt')
        test_data = torch.load(datasets_path / 'test.pt')

        # Mock datamodule
        self.datamodule.train_data, self.datamodule.val_data, self.datamodule.test_data = train_data, val_data, test_data
        self.datamodule.data = test_data
        outputs = {}

        # Embeddings
        Z_raw = np.load(output_path / 'embeddings.npy', allow_pickle=True)
        if Z_raw.dtype == np.object:
            Z_dict = {
                k: torch.from_numpy(v).float()
                for k, v in Z_raw.item().items()
            }
            outputs['Z'] = torch.cat(list(Z_dict.values()), dim=0)
            outputs['X'] = outputs['Z']
        else:
            outputs['Z'] = torch.from_numpy(Z_raw).float()
            outputs['X'] = outputs['Z']

        if (output_path / 'means.npy').exists():
            mus = torch.from_numpy(np.load(output_path / 'means.npy', allow_pickle=True)).float()
        else:
            mus = None
        outputs['mus'] = mus

        if (output_path / 'assignments.npy').exists():
            z_raw = np.load(output_path / 'assignments.npy', allow_pickle=True)
            if z_raw.dtype == np.object:
                z_dict = {
                    k: torch.from_numpy(v).long()
                    for k, v in z_raw.item().items()
                }
                outputs['z'] = torch.cat(list(z_dict.values()), dim=0)
            else:
                outputs['z'] = torch.from_numpy(z_raw).long()
        else:
            outputs['z'] = None

        # Mock model
        self.model.trainer = self.trainer
        self.trainer.strategy._model = self.model

        # Mock callbacks
        progress_bar = next(c for c in self.trainer.callbacks if isinstance(c, ProgressBarBase))
        progress_bar._trainer = self.trainer

        for callback in self.trainer.callbacks:
            if hasattr(callback, 'hparams') and hasattr(callback.hparams, 'interval'):
                callback.hparams.interval = 1

        # Mock test
        mock_evaluation_epoch(self.trainer, self.model, OutputExtractor([outputs]), mode=EvaluationMode.VALIDATION)
        mock_evaluation_epoch(self.trainer, self.model, OutputExtractor([outputs]), mode=EvaluationMode.TEST)

        # TODO: Mock prediction?

    def tags(self):
        tags = super().tags()
        if self.args.baseline:
            tags.append('baseline')

        return tags


if __name__ == '__main__':
    # args = parse_args(Args)[0]
    # run(args)
    EvaluateExecutor().cli()
