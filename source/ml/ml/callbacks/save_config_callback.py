from pathlib import Path

import yaml
from pytorch_lightning import Callback, Trainer, LightningModule
from simple_parsing import Serializable

from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveConfigCallback(Callback):
    config: Serializable

    def __init__(self, config: Serializable, log: bool = True) -> None:
        super().__init__()
        self.config = config
        self.log = log

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log:
            logger.info("=" * 80)
            logger.info(f"Current config:")
            config_str = self.config.dumps_yaml(Dumper=MyDumper)

            logger.info(f"\n{config_str}")
            logger.info("=" * 80)

        for train_logger in trainer.loggers:
            if train_logger.save_dir is not None:
                save_dir = Path(train_logger.save_dir)
                logger.info(f"Saving config in: {save_dir}")
                self.config.save_yaml(save_dir / 'config.yaml', Dumper=MyDumper, default_flow_style=False)


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

    def ignore_aliases(self, data):
        return True
