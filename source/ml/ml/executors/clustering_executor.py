from dataclasses import dataclass
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets.utils.base import DATASET_REGISTRY, GraphDataset
from ml.callbacks.clustering_monitor import ClusteringMonitor
from ml.callbacks.gmm_visualizer_callback import GMMVisualizerCallback
from ml.data import ConcatDataset, PretrainedEmbeddingsDataset
from ml.models.dpm_clustering import DPMClusteringModelParams, DPMClusteringModel
from ml.utils.config import TrainerParams, dataset_choices
from shared import parse_args, get_logger, RESULTS_PATH

EXECUTOR_NAME = 'clustering_train_executor'
TASK_NAME = 'clustering'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args:
    dataset: str = dataset_choices()
    pretrained_path: List[str] = None
    hparams: DPMClusteringModelParams = DPMClusteringModelParams()
    trainer: TrainerParams = TrainerParams()


def train(args: Args):
    graph_dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    dataset = ConcatDataset([
        PretrainedEmbeddingsDataset.from_pretrained(path)
        for path in args.pretrained_path
    ])
    repr_dim = dataset[[0]].shape[-1]
    model = DPMClusteringModel(dataset, hparams=args.hparams, repr_dim=repr_dim)

    run_name = f'{TASK_NAME}/{graph_dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ClusteringMonitor(),
        # ClusteringVisualizerCallback(),
        GMMVisualizerCallback(),
    ]

    logger.info('Training model')
    trainer = Trainer(**args.trainer.to_dict(), default_root_dir=str(root_dir), callbacks=callbacks)
    trainer.fit(model, model.train_dataloader())


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
