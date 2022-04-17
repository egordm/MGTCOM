from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from ml.utils import Metric, OutputExtractor

EPS = 1e-15


class Node2VecModel(pl.LightningModule):
    def __init__(
            self,
            embedder: torch.nn.Module,
            metric: Metric = Metric.L2,
            hparams: Any = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.embedder = embedder
        self.metric = metric
        self.sim_fn = metric.pairwise_sim_fn

    def forward(self, X: Any) -> Any:
        return self.embedder(X)

    def loss(self, batch):
        pos_walks, neg_walks, node_ids = batch

        Z = self.embedder(node_ids)

        pos_walks_Z = Z[pos_walks.view(-1)].view(*pos_walks.shape, Z.shape[-1])
        p_head, p_rest = pos_walks_Z[:, 0].unsqueeze(dim=1), pos_walks_Z[:, 1:]
        p_sim = self.sim_fn(p_head, p_rest).view(-1)
        pos_loss = -torch.log(torch.sigmoid(p_sim) + EPS).mean()

        neg_walks_Z = Z[neg_walks.view(-1)].view(*neg_walks.shape, Z.shape[-1])
        n_head, n_rest = neg_walks_Z[:, 0].unsqueeze(dim=1), neg_walks_Z[:, 1:]
        n_sim = self.sim_fn(n_head, n_rest).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(n_sim) + EPS).mean()

        return pos_loss + neg_loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch
        return self.forward(X)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = OutputExtractor(outputs)
        epoch_loss = outputs.extract_mean('loss')
        self.log('epoch_loss', epoch_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
