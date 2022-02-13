import argparse
from typing import Any

import torch
import torchmetrics
from torch.nn import ReLU

from torch_geometric.nn import SAGEConv, Sequential, to_hetero

import pytorch_lightning as pl

import ml
from ml import StarWars, BaseModule

parser = argparse.ArgumentParser()
parser.add_argument('--use_hgt_loader', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = StarWars()
data = dataset[0].to(device, 'x', 'y')

embedding_dim = 32
node_type = 'Character'
data_module = ml.EdgeLoaderDataModule(
    data,
    batch_size=16, num_neighbors=[4] * 2, num_workers=8, node_type=node_type, neg_sample_ratio=1
)
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

embedding_model = Sequential('x, edge_index', [
    (SAGEConv((-1, -1), embedding_dim), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (SAGEConv((-1, -1), embedding_dim), 'x, edge_index -> x'),
])
embedding_model = to_hetero(embedding_model, data.metadata(), aggr='mean')


class Net(BaseModule):
    def __init__(
            self,
            embedding_model, node_type,
            *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.node_type = node_type

        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.lin = torch.nn.Linear(1, 2)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def _step(self, batch: Any) -> dict:
        batch_l, batch_r, label = batch
        batch_size = batch_l[self.node_type].batch_size

        emb_l = self.embedding_model(batch_l.x_dict, batch_l.edge_index_dict)[node_type][:batch_size]
        emb_r = self.embedding_model(batch_r.x_dict, batch_r.edge_index_dict)[node_type][:batch_size]

        sim = self.sim(emb_l, emb_r)
        out = self.lin(torch.unsqueeze(sim, 1))
        loss = self.ce_loss(out, label)

        pred = out.argmax(dim=-1).detach()
        return {
            'loss': loss,
            'accuracy': (pred, label),
        }

    def training_step(self, batch):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def configure_metrics(self):
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
        }


model = Net(embedding_model, node_type)
trainer = pl.Trainer(
    gpus=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    ],
    max_epochs=20,
    enable_model_summary=True,
)
trainer.fit(model, data_module)
u = 0