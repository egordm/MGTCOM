from typing import Dict, Union, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torch_geometric.typing import NodeType

from ml.utils import OutputExtractor, OptimizerParams


class BaseEmbeddingModel(pl.LightningModule):
    hparams: OptimizerParams
    val_Z_dict: Dict[NodeType, Tensor] = None
    val_Z: Tensor = None

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        outputs = OutputExtractor(outputs)
        self.val_Z_dict = outputs.extract_cat_dict('Z_dict')
        self.val_Z = torch.cat(list(self.val_Z_dict.values()), dim=0)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return dict(
            Z_dict=self.forward(batch)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
