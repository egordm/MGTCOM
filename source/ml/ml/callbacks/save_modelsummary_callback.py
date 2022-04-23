from pathlib import Path

import wandb
from pytorch_lightning import Callback, Trainer, LightningModule
from torchinfo import summary
from torchinfo.formatting import FormattingOptions

from shared import get_logger

logger = get_logger(Path(__file__).stem)


class SaveModelSummaryCallback(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        info = summary(pl_module, depth=8, verbose=1)

        info.formatting = FormattingOptions(
            max_depth=8, verbose=2,
            col_names=info.formatting.col_names,
            col_width=info.formatting.col_width,
            row_settings=info.formatting.row_settings,
        )

        save_dir = Path(wandb.run.dir) / 'model_info.txt'
        save_dir.write_text(str(info))
