import torch
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from datasets import GraphDataset


class BaseExecutor(LightningCLI):
    @property
    def dataset(self) -> GraphDataset:
        return self.config_init['dataset']
