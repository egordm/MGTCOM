from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn.functional as F
from torch_geometric.typing import Metadata
from torch_geometric.nn import HGTConv, Linear
import torchmetrics
import pytorch_lightning as pl
import math

from shared.constants import BENCHMARKS_RESULTS
from shared.graph import DataGraph
from benchmarks.evaluation import get_metric_list
from shared.schema import GraphSchema, DatasetSchema
from shared.graph import CommunityAssignment
import pandas as pd
from datasets.scripts import export_to_visualization

import ml
from ml.data.datasets import StarWars

dataset = StarWars()
data = dataset[0]
data

data_module = ml.EdgeLoaderDataModule(
    data, batch_size=16, num_samples=[4] * 2, num_workers=8, node_type='Character', neg_sample_ratio=1
)


class HGTModule(torch.nn.Module):
    def __init__(
            self,
            node_type,
            metadata: Metadata,
            hidden_channels=64,
            num_heads=2,
            num_layers=1
    ):
        super().__init__()
        self.node_type = node_type
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict[self.node_type]


class ClusteringModule(torch.nn.Module):
    def __init__(
            self, rep_dim: int, n_clusters: int,
            cluster_centers: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.rep_dim = rep_dim

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters, self.rep_dim, dtype=torch.float
            )
            torch.nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            assert cluster_centers.shape == (self.n_clusters, self.rep_dim)
            initial_cluster_centers = cluster_centers
        self.cluster_centers = torch.nn.Parameter(initial_cluster_centers)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, batch: torch.Tensor):
        sim = torch.cdist(batch, self.cluster_centers, p=2)
        return self.activation(sim)

    def forward_assign(self, batch: torch.Tensor):
        q = self(batch)
        return q.argmax(dim=1)


class LinkPredictionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        return self.ce_loss(pred, label)


class ClusteringLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def forward(self, q_l: torch.Tensor, q_r: torch.Tensor, label: torch.Tensor):
        p = torch.sum(q_l, dim=0) + torch.sum(q_r, dim=0)
        p /= p.sum()
        ne = math.log(p.size(0)) + (p * torch.log(p)).sum()

        sim = (self.cos_sim(q_l, q_r) + 1) / 2
        pred = torch.stack([1 - sim, sim], dim=1)
        loss = self.ce_loss(pred, label)

        return loss, ne


class Net(ml.BaseModule):
    def __init__(
            self,
            embedding_module: HGTModule,
            clustering_module: ClusteringModule,
    ):
        super().__init__()
        self.embedding_module = embedding_module
        self.clustering_module = clustering_module
        self.dist = torch.nn.PairwiseDistance(p=2)
        self.lin = torch.nn.Linear(1, 2)

        self.link_prediction_loss = LinkPredictionLoss()
        self.clustering_loss = ClusteringLoss()

        self.is_pretraining = True

    def configure_metrics(self) -> Dict[str, Tuple[torchmetrics.Metric, bool]]:
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'hp_loss': (torchmetrics.MeanMetric(), True),
            'cc_loss': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
            'ne': (torchmetrics.MeanMetric(), True),
        }

    def _step(self, batch: torch.Tensor):
        batch_l, batch_r, label = batch
        batch_size = batch_l[self.embedding_module.node_type].batch_size

        emb_l = self.embedding_module(batch_l.x_dict, batch_l.edge_index_dict)[:batch_size]
        emb_r = self.embedding_module(batch_r.x_dict, batch_r.edge_index_dict)[:batch_size]
        dist = self.dist(emb_l, emb_r)
        out = self.lin(torch.unsqueeze(dist, 1))
        hp_loss = self.link_prediction_loss(out, label)

        out_dict = {}
        if self.is_pretraining:
            loss = hp_loss
        else:
            q_l = self.clustering_module(emb_l)
            q_r = self.clustering_module(emb_r)
            cc_loss, ne = self.clustering_loss(q_l, q_r, label)
            loss = hp_loss + 2 * cc_loss + ne * 0.01
            out_dict['ne'] = ne.detach()
            out_dict['cc_loss'] = cc_loss.detach()

        pred = out.argmax(dim=-1).detach()
        return {
            'loss': loss,
            'hp_loss': hp_loss.detach(),
            'accuracy': (pred, label),
            **out_dict
        }

    def forward(self, batch):
        batch_size = batch[self.embedding_module.node_type].batch_size
        emb = self.embedding_module(batch.x_dict, batch.edge_index_dict)[:batch_size]
        q = self.clustering_module(emb)
        return emb, q

    def training_step(self, batch):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


embedding_module = HGTModule(node_type='Character', metadata=data.metadata(), hidden_channels=32, num_heads=2,
                             num_layers=2)
clustering_module = ClusteringModule(rep_dim=32, n_clusters=5)
model = Net(embedding_module, clustering_module)

model.is_pretraining = True
trainer = pl.Trainer(
    gpus=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    ],
    max_epochs=20,
    enable_model_summary=True,
    # logger=wandb_logger
)
trainer.fit(model, data_module)

model.is_pretraining = False
trainer = pl.Trainer(
    gpus=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    ],
    max_epochs=20,
    enable_model_summary=True,
    # logger=wandb_logger
)
trainer.fit(model, data_module)

save_path = BENCHMARKS_RESULTS.joinpath('analysis', 'pyg-hgt-comopt-contrastive')
save_path.mkdir(parents=True, exist_ok=True)

predictions = trainer.predict(model, data_module)
embeddings, assignments = map(lambda x: torch.cat(x, dim=0).detach().cpu(), zip(*predictions))
assignments = torch.argmax(assignments, dim=1)

labeling = pd.Series(assignments.squeeze(), index=dataset.node_mapping(), name="cid")
labeling.index.name = "nid"
comlist = CommunityAssignment(labeling)
comlist.save_comlist(save_path.joinpath('schema.comlist'))

export_to_visualization.run(
    export_to_visualization.Args(
        dataset='star-wars',
        version='base',
        run_paths=[str(save_path)]
    )
)

# Calculate Evaluation Metrics
DATASET = DatasetSchema.load_schema('star-wars')
schema = GraphSchema.from_dataset(DATASET)
G = DataGraph.from_schema(schema)

metrics = get_metric_list(ground_truth=False, overlapping=False)

results = pd.DataFrame([
    {
        'metric': metric_cls.metric_name(),
        'value': metric_cls.calculate(G, comlist)
    }
    for metric_cls in metrics]
)
results
