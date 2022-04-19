from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.clustering_visualizer_callback import ClusteringVisualizerCallback
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.callbacks.progress_bar import CustomProgressBar
from ml.callbacks.save_embeddings_callback import SaveEmbeddingsCallback
from ml.data import PretrainedEmbeddingsDataset, SyntheticGMMDataset
from ml.models.dpmmsc import DPMMSCModelParams, DPMMSubClusteringModel
from ml.models.mgcom_combi import MGCOMCombiDataModule, MGCOMCombiDataModuleParams, \
    MGCOMCombiModel
from ml.models.mgcom_comdet import MGCOMComDetDataModuleParams, MGCOMComDetDataModule
from ml.utils import dataset_choices, DataLoaderParams, OptimizerParams, TrainerParams
from ml.utils.labelling import extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from shared import get_logger, parse_args, RESULTS_PATH

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'community_detection'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args:
    dataset: str = dataset_choices()
    pretrained_path: Optional[str] = None
    hparams: DPMMSCModelParams = DPMMSCModelParams()
    data_params: MGCOMComDetDataModuleParams = MGCOMComDetDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams(batch_size=200)
    trainer_params: TrainerParams = TrainerParams(max_epochs=200)


def train(args: Args):
    graph_dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    if args.pretrained_path:
        logger.info(f'Using pretrained embeddings from {args.pretrained_path}')
        dataset = PretrainedEmbeddingsDataset.from_pretrained(args.pretrained_path, graph_dataset.name)
    else:
        logger.info('No pretrained embeddings provided, using synthetic dataset')
        dataset = SyntheticGMMDataset()

    data_module = MGCOMComDetDataModule(
        dataset=dataset,
        hparams=args.data_params,
        loader_params=args.loader_params,
    )

    model = DPMMSubClusteringModel(
        dataset.repr_dim,
        hparams=args.hparams,
    )

    run_name = f'{TASK_NAME}/{dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        CustomProgressBar(),
        LearningRateMonitor(logging_interval='step'),
        ClusteringVisualizerCallback(),
        SaveEmbeddingsCallback(),
    ]

    logger.info('Training model')
    trainer = Trainer(
        **args.trainer_params.to_dict(),
        default_root_dir=str(root_dir),
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    trainer.predict(model, data_module)


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
