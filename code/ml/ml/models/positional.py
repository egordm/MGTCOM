from typing import Any, List

import pytorch_lightning as pl
import torch.nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import HeteroData

from ml.layers import HingeLoss, NegativeEntropyRegularizer


class PositionalModel(pl.LightningModule):
    def __init__(
            self,
            embedding_module: torch.nn.Module,
            clustering_module: torch.nn.Module,
            ne_weight: float = 0.001,
            use_clustering: bool = False,
            *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_module = embedding_module
        self.clustering_module = clustering_module

        self.ne_weight = ne_weight
        self.use_clustering = use_clustering

        self.positional_loss = HingeLoss()
        self.clustering_loss = HingeLoss()
        self.ne_loss = NegativeEntropyRegularizer()

    def forward(self, batch: HeteroData, *args, **kwargs) -> Any:
        return self.embedding_module(batch, *args, **kwargs)

    def step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        sample_data, pn_data, node_types = batch

        p_emb_dict = self.embedding_module(sample_data)
        p_emb = torch.cat([p_emb_dict[node_type] for node_type in node_types], dim=0)

        p_loss = self.positional_loss(p_emb, pn_data)

        if self.use_clustering:
            c_emb = self.clustering_module(p_emb)

            c_loss = self.clustering_loss(c_emb, pn_data)
            ne = self.ne_loss(c_emb)

            return {
                'loss': p_loss + c_loss + (ne * self.ne_weight),
                'p_loss': p_loss.detach(),
                'c_loss': c_loss.detach(),
                'ne_loss': ne.detach(),
            }
        else:
            return {
                'loss': p_loss,
                'p_loss': p_loss.detach(),
            }

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self.step(*args, **kwargs)

    def on_train_batch_end(self, outputs, *args, **kwargs) -> None:
        log_data = {
            k: v.detach() if isinstance(v, Tensor) else v
            for k, v in outputs.items()
        }
        self.log_dict(log_data, prog_bar=True)
        self.log('pre', 1.0 if self.use_clustering else 0.0, prog_bar=True)
        super().on_train_batch_end(outputs, *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        super().on_predict_epoch_end(results)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return {
            'optimizer': optimizer,
        }




