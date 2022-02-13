import torch
import torchmetrics

import ml
from experiments.models.embedding import LinkPredictionModule


class LinkPredictionNet(ml.BaseModule):
    def __init__(self, predictor: LinkPredictionModule, params: dict = None) -> None:
        super().__init__()
        self.predictor = predictor
        self.params = params or {}

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: torch.Tensor):
        return self.predictor(batch)

    def _step(self, batch: torch.Tensor):
        _, _, label = batch
        logits, dist, emb_l, emb_r = self.forward(batch)
        loss = self.ce_loss(logits, label)

        pred = logits.argmax(dim=-1)
        return {
            'loss': loss,
            'accuracy': (pred, label),
        }

    def training_step(self, batch):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def configure_metrics(self):
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.get('lr', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
