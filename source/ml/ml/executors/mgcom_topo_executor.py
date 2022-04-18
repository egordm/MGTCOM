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

    # Dataset dependent things
    G, _, _, node_offsets = igraph_from_hetero(data_module.val_data)
    comm = G.community_multilevel()

    callbacks = [
        CustomProgressBar(),
        LearningRateMonitor(logging_interval='step'),
        EmbeddingVisualizerCallback(
            node_labels={
                'Louvain Labels': torch.tensor(comm.membership, dtype=torch.long),
            }
        )
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
