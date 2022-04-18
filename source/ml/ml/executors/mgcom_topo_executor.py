from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from datasets.utils.conversion import igraph_from_hetero
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.callbacks.progress_bar import CustomProgressBar
from ml.models.mgcom_topo import MGCOMTopoModelParams, MGCOMTopoDataModuleParams, MGCOMTopoDataModule, MGCOMTopoModel
from ml.utils import dataset_choices, DataLoaderParams, OptimizerParams, TrainerParams
from ml.utils.labelling import extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from shared import get_logger, parse_args, RESULTS_PATH

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'embedding_topo'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args:
    dataset: str = dataset_choices()
    hparams: MGCOMTopoModelParams = MGCOMTopoModelParams()
    data_params: MGCOMTopoDataModuleParams = MGCOMTopoDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    trainer_params: TrainerParams = TrainerParams()


def train(args: Args):
    dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()

    data_module = MGCOMTopoDataModule(
        dataset=dataset,
        hparams=args.data_params,
        loader_params=args.loader_params,
    )

    model = MGCOMTopoModel(
        data_module.metadata, data_module.num_nodes_dict,
        hparams=args.hparams,
        optimizer_params=args.optimizer_params,
    )

    run_name = f'{TASK_NAME}/{dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    logger.info('Extracting labels for visualization')
    node_labels = {}
    node_labels['Louvain Labels'] = extract_louvain_labels(data_module.val_data)
    if isinstance(dataset, GraphDataset) and dataset.snapshots is not None:
        node_timestamps = extract_timestamp_labels(data_module.val_data)
        for i, snapshot in dataset.snapshots.items():
            snapshot_labels = extract_snapshot_labels(node_timestamps, snapshot)
            node_labels[f'{i} Temporal Snapshots'] = snapshot_labels

    callbacks = [
        CustomProgressBar(),
        LearningRateMonitor(logging_interval='step'),
        EmbeddingVisualizerCallback(node_labels=node_labels)
    ]

    logger.info('Training model')
    trainer = Trainer(
        **args.trainer_params.to_dict(),
        default_root_dir=str(root_dir),
        callbacks=callbacks,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)