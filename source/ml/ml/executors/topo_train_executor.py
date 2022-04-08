from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from datasets.utils.base import DATASET_REGISTRY
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
    dataset = DATASET_REGISTRY[args.dataset]()
    model = TopoEmbeddingModel(dataset, hparams=args.hparams)

    run_name = f'topo/{dataset.name}'
    root_dir = RESULTS_PATH / run_name
    root_dir.mkdir(exist_ok=True, parents=True)

    logger.info('Training model')
    trainer = Trainer(**args.trainer.to_dict(), default_root_dir=str(root_dir))
    trainer.fit(model, model.train_dataloader())

    logger.info('Computing embeddings')
    pred = trainer.predict(model, model.predict_dataloader())
    emb = merge_dicts(pred, lambda xs: torch.cat(xs, dim=0))

    logger.info('Saving embeddings')
    torch.save(emb, Path(trainer.log_dir) / 'embeddings.pt')




if __name__ == '__main__':
    args: Args = parse_args(Args)[0]
    train(args)
