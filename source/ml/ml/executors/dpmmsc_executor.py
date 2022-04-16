from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets.utils.base import DATASET_REGISTRY, GraphDataset
from ml.callbacks.clustering_monitor import ClusteringMonitor
from ml.callbacks.dpmm_visualizer_callback import DPMMVisualizerCallback
from ml.callbacks.gmm_visualizer_callback import GMMVisualizerCallback
from ml.data import ConcatDataset, PretrainedEmbeddingsDataset
from ml.data.generated.synthetic_gmm import SyntheticGMMDataset
from ml.models.dpmmsc import DPMMSubClusteringModel, DPMMSCModelParams
from ml.utils.config import TrainerParams, dataset_choices
from shared import parse_args, get_logger, RESULTS_PATH

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'clustering'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args:
    dataset: str = dataset_choices()
    pretrained_path: Optional[List[str]] = None
    hparams: DPMMSCModelParams = DPMMSCModelParams()
    trainer: TrainerParams = TrainerParams()


def train(args: Args):
    graph_dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    if args.pretrained_path:
        logger.info(f'Using pretrained embeddings from {args.pretrained_path}')
        dataset = ConcatDataset([
            PretrainedEmbeddingsDataset.from_pretrained(path)
            for path in args.pretrained_path
        ])
    else:
        logger.info('No pretrained embeddings provided, using synthetic dataset')
        dataset = ConcatDataset([
            SyntheticGMMDataset()
        ])

    model = DPMMSubClusteringModel(dataset, hparams=args.hparams)

    run_name = f'{TASK_NAME}/{graph_dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        # ClusteringMonitor(),
        # ClusteringVisualizerCallback(),
        DPMMVisualizerCallback(logging_interval=3),
        # GMMVisualizerCallback(logging_interval=1),
    ]

    logger.info('Training model')
    trainer = Trainer(**args.trainer.to_dict(), default_root_dir=str(root_dir), callbacks=callbacks, num_sanity_val_steps=0)
    trainer.fit(model, model.train_dataloader())


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
