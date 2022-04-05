import logging
from pathlib import Path

import torch
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from ml.datasets import GraphDataset
from ml.layers import BaseInitializer


class BaseExecutor(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--load_dir", type=str)

    @property
    def dataset(self) -> GraphDataset:
        return self.config_init['dataset']

    @property
    def loader(self):
        return self.config_init['loader']

    @property
    def initializer(self) -> BaseInitializer:
        return self.config_init['initializer']

    @property
    def repr_dim(self):
        return self.config['repr_dim']

    def compute_embeddings(self, trainer=None):
        loader = self.loader
        model = self.model
        trainer = trainer or self.trainer

        embeddings = model.compute_embeddings(trainer, loader)
        return embeddings

    def compute_assignments(self, emb, trainer=None):
        model = self.model
        emb = emb or self.compute_embeddings(trainer)
        return {k: model.compute_assignments(emb).detach().cpu() for k, emb in emb.items()}

    def load(self, checkpoint_path: str):
        load_state = torch.load(checkpoint_path)
        self.load_state = load_state

        self.pre_load(load_state)

        # Handle case when checkpoint is loaded after initialization
        if hasattr(self, 'config_init'):
            logging.warning('Dont load the checkpoint after initialization')
            self.post_load(load_state)

    def pre_load(self, load_state):
        pass

    def post_load(self, load_state):
        state = self.model.state_dict()
        state.update(load_state)
        self.model.load_state_dict(state)

    def before_instantiate_classes(self) -> None:
        if 'load_dir' in self.config:
            self.load(str(Path(self.config['load_dir']).joinpath('model.pt')))

        super().before_instantiate_classes()

    def instantiate_classes(self) -> None:
        super().instantiate_classes()

        # Handle case when checkpoint is loaded before initialization
        if hasattr(self, 'load_state'):
            self.post_load(self.load_state)


