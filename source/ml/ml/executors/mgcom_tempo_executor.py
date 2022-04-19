from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from simple_parsing import Serializable

from datasets import GraphDataset
from datasets.utils.base import DATASET_REGISTRY
from ml.callbacks.embedding_visualizer_callback import EmbeddingVisualizerCallback
from ml.callbacks.progress_bar import CustomProgressBar
from ml.callbacks.save_config_callback import SaveConfigCallback
from ml.callbacks.save_graph_callback import SaveGraphCallback
from ml.models.mgcom_feat import MGCOMFeatModelParams, MGCOMTempoDataModuleParams, MGCOMTempoDataModule, MGCOMFeatModel
from ml.utils import dataset_choices, DataLoaderParams, OptimizerParams, TrainerParams, HParams
from ml.utils.labelling import extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from shared import get_logger, parse_args, RESULTS_PATH

EXECUTOR_NAME = Path(__file__).stem
TASK_NAME = 'embedding_tempo'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args(Serializable):
    dataset: str = dataset_choices()
    hparams: MGCOMFeatModelParams = MGCOMFeatModelParams()
    data_params: MGCOMTempoDataModuleParams = MGCOMTempoDataModuleParams()
    loader_params: DataLoaderParams = DataLoaderParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    trainer_params: TrainerParams = TrainerParams()


def train(args: Args):
    dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()

    data_module = MGCOMTempoDataModule(
        dataset=dataset,
        hparams=args.data_params,
        loader_params=args.loader_params,
    )

    model = MGCOMFeatModel(
        data_module.metadata, data_module.num_nodes_dict,
        hparams=args.hparams,
        optimizer_params=args.optimizer_params,
    )

    run_name = f'{TASK_NAME}/{dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    logger.info('Extracting labels for visualization')
    node_labels = {}
    node_labels['Louvain Labels'] = extract_louvain_labels(data_module.data)
    if isinstance(dataset, GraphDataset) and dataset.snapshots is not None:
        node_timestamps = extract_timestamp_labels(data_module.data)
        for i, snapshot in dataset.snapshots.items():
            snapshot_labels = extract_snapshot_labels(node_timestamps, snapshot)
            node_labels[f'{i} Temporal Snapshots'] = snapshot_labels

    val_node_labels = {
        label_name: {
            node_type: label_dict[node_type][perm]
            for node_type, perm in data_module.val_data.id_dict.items()
        }
        for label_name, label_dict in node_labels.items()
    }

    callbacks = [
        CustomProgressBar(),
        LearningRateMonitor(logging_interval='step'),
        EmbeddingVisualizerCallback(val_node_labels=val_node_labels),
        SaveConfigCallback(args),
        SaveGraphCallback(data_module.data, node_labels=node_labels),
    ]

    logger.info('Training model')
    trainer = Trainer(
        **args.trainer_params.to_dict(),
        default_root_dir=str(root_dir),
        callbacks=callbacks,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    trainer.predict(model, data_module)


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
