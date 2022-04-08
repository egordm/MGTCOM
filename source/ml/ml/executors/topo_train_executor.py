from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.base import DummyLogger

from datasets.utils.base import DATASET_REGISTRY, GraphDataset
from datasets.utils.conversion import igraph_from_hetero
from ml.callbacks.embedding_collector import EmbeddingsCollectorCallback
from ml.callbacks.modularity_monitor import ModularityMonitorCallback
from ml.layers.clustering import KMeans
from ml.layers.metrics.modularity import newman_girvan_modularity
from ml.models.topo_embedding import TopoEmbeddingModel, TopoEmbeddingModelParams
from ml.utils.config import TrainerParams, dataset_choices
from ml.utils.dict import merge_dicts
from shared.cli import parse_args
from shared.logger import get_logger
from shared.paths import RESULTS_PATH

logger = get_logger('TopoTrainExecutor')


@dataclass
class Args:
    dataset: str = dataset_choices()
    hparams: TopoEmbeddingModelParams = TopoEmbeddingModelParams()
    trainer: TrainerParams = TrainerParams()


def train(args: Args):
    dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    model = TopoEmbeddingModel(dataset, hparams=args.hparams)

    run_name = f'topo/{dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        EmbeddingsCollectorCallback(),
        ModularityMonitorCallback(),
    ]

    logger.info('Training model')
    trainer = Trainer(**args.trainer.to_dict(), default_root_dir=str(root_dir), callbacks=callbacks)
    trainer.fit(model, model.train_dataloader())

    logger.info('Computing embeddings')
    pred = trainer.predict(model, model.predict_dataloader())
    emb_dict = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

    save_dir = Path(trainer.log_dir)
    logger.info('Saving embeddings')
    torch.save(emb_dict, save_dir / 'embeddings.pt')

    G, _, _, _ = igraph_from_hetero(dataset.data, node_attrs=dict(label=dataset.data.name_dict))
    G.write_graphml(str(save_dir / 'graph.graphml'))


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
