from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets.utils.base import DATASET_REGISTRY, GraphDataset
from datasets.utils.conversion import igraph_from_hetero
from ml.callbacks import PreclusteringMonitor, EmbeddingMonitor
from ml.models.topo_embedding import TopoEmbeddingModel, TopoEmbeddingModelParams
from ml.utils.config import TrainerParams, dataset_choices
from ml.utils.dict import merge_dicts
from shared.cli import parse_args
from shared.logger import get_logger
from shared.paths import RESULTS_PATH

EXECUTOR_NAME = 'embedding_topo_executor'
TASK_NAME = 'topo'

logger = get_logger(EXECUTOR_NAME)


@dataclass
class Args:
    dataset: str = dataset_choices()
    hparams: TopoEmbeddingModelParams = TopoEmbeddingModelParams()
    trainer: TrainerParams = TrainerParams()


def train(args: Args):
    dataset: GraphDataset = DATASET_REGISTRY[args.dataset]()
    model = TopoEmbeddingModel(dataset, hparams=args.hparams)

    root_dir = RESULTS_PATH / TASK_NAME / dataset.name
    root_dir.mkdir(exist_ok=True, parents=True)

    wandb_logger = WandbLogger(
        save_dir=str(root_dir),
        config={
            'dataset': dataset.name,
            'trainer_params': args.trainer.to_dict(),
        },
        tags=[TASK_NAME, dataset.name],
        job_type=TASK_NAME,
    )

    preclustering_monitor = PreclusteringMonitor()
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        preclustering_monitor,
        EmbeddingMonitor()
    ]

    logger.info('Training model')
    trainer = Trainer(**args.trainer.to_dict(), callbacks=callbacks, logger=wandb_logger)
    trainer.fit(model, model.train_dataloader())

    logger.info('Computing embeddings')
    pred = trainer.predict(model, model.predict_dataloader())
    emb_dict = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

    save_dir = Path(wandb.run.dir)
    logger.info('Saving embeddings')
    torch.save(emb_dict, save_dir / 'embeddings.pt')

    logger.info('Saving graph and preclustering')
    G, _, _, _ = igraph_from_hetero(dataset.data, node_attrs=dict(label=dataset.data.name_dict))
    _, I_flat = preclustering_monitor.run_preclutering(emb_dict)
    G.vs['I'] = I_flat.numpy()
    G.write_graphml(str(save_dir / 'graph.graphml'))


if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
