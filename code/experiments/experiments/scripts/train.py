from typing import List

import pytorch_lightning as pl
import torch
import torchmetrics
import pandas as pd
from torch_geometric.data import HeteroData

from experiments import ClusteringModule, ClusterCohesionLoss, NegativeEntropyRegularizer, cosine_cdist
from shared.constants import TMP_PATH, BENCHMARKS_RESULTS
from shared.graph import CommunityAssignment
import ml
import experiments

node_type = 'Character'
repr_dim = 32
save_path = TMP_PATH.joinpath('pyg-sage-comopt')
callbacks = [
    pl.callbacks.ModelSummary(),
    pl.callbacks.LearningRateMonitor(),
    pl.callbacks.EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=True, mode="min")
]

dataset = ml.StarWars()
data = dataset[0]
data_module = ml.EdgeLoaderDataModule(
    data,
    batch_size=16, num_samples=[4] * 2,
    num_workers=0, node_type=node_type, neg_sample_ratio=1
)

G = dataset.G
G.to_undirected()
initial_clustering = G.community_multilevel()

embedding_module = experiments.GraphSAGEModule(node_type, data.metadata(), repr_dim, n_layers=2)

trainer = pl.Trainer(callbacks=callbacks, max_epochs=20, default_root_dir=str(save_path))

if save_path.joinpath('pretrained.ckpt').exists():
    print('Loading pretrained model')
    pretrain_model = experiments.LinkPredictionModule.load_from_checkpoint(
        save_path.joinpath('pretrained.ckpt'),
        embedding_module=embedding_module
    )
else:
    pretrain_model = experiments.LinkPredictionModule(embedding_module, dist='cosine')
    trainer.fit(pretrain_model, data_module)
    trainer.save_checkpoint(str(save_path.joinpath('pretrained.ckpt')))



# initializer = experiments.KMeansInitializer(repr_dim=32, k=5, niter=20, verbose=True, nredo=5)
# assignment, centroids = initializer.fit(embeddings)

embeddings = torch.cat(trainer.predict(embedding_module, data_module), dim=0).detach().cpu()
assignment = torch.tensor(initial_clustering.membership)
cluster_count = len(initial_clustering)
assigned_count = torch.zeros(cluster_count, dtype=torch.long).scatter_add(0, assignment, torch.ones_like(assignment, dtype=torch.long))
centroids = torch.zeros(cluster_count, repr_dim, dtype=torch.float).index_add_(0, assignment, embeddings)
centroids = centroids / assigned_count.unsqueeze(1)

clustering_module = experiments.ExplicitClusteringModule(
    repr_dim=repr_dim, n_clusters=cluster_count, centroids=torch.tensor(centroids), dist='cosine'
)
# clustering_module = experiments.ImplicitClusteringModule(
#     repr_dim=repr_dim, n_clusters=5,
# )


class SelfSupervisedClusteringNet(ml.BaseModule):
    def __init__(
            self,
            predictor_module: experiments.LinkPredictionModule,
            clustering_module: ClusteringModule,
            params: dict = None
    ) -> None:
        super().__init__()
        self.predictor_module = predictor_module
        self.clustering_module = clustering_module
        self.params = params or {}

        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.cc_loss_fn = ClusterCohesionLoss()
        self.ne_loss_fn = NegativeEntropyRegularizer()

    def forward(self, batch: torch.Tensor):
        emb = self.predictor_module.embedding_module(batch)
        q = self.clustering_module(emb)
        return emb, q

    def _step(self, batch: torch.Tensor):
        _, _, label = batch
        logits, dist, emb_l, emb_r = self.predictor_module(batch)
        he_loss = self.ce_loss_fn(logits, label)

        q_l = self.clustering_module(emb_l)
        q_r = self.clustering_module(emb_r)
        cc_loss = self.cc_loss_fn(q_l, q_r, label)
        ne = self.ne_loss_fn(q_l, q_r)

        loss = he_loss + cc_loss * self.params.get('cc_weight', 2.0) + ne * self.params.get('ne_weight', 0.01)

        pred = logits.argmax(dim=-1)
        return {
            'loss': loss,
            'he_loss': he_loss.detach(),
            'cc_loss': cc_loss.detach(),
            'ne': ne.detach(),
            'accuracy': (pred.detach(), label),
        }

    def configure_metrics(self):
        return {
            'loss': (torchmetrics.MeanMetric(), True),
            'he_loss': (torchmetrics.MeanMetric(), True),
            'cc_loss': (torchmetrics.MeanMetric(), True),
            'ne': (torchmetrics.MeanMetric(), True),
            'accuracy': (torchmetrics.Accuracy(), True),
        }


# class FullClusteringNet(ml.BaseModule):
#     def __init__(
#             self,
#             embedding_module: experiments.EmbeddingModule,
#             clustering_module: ClusteringModule,
#             params: dict = None
#     ) -> None:
#         super().__init__()
#         self.embedding_module = embedding_module
#         self.clustering_module = clustering_module
#         self.params = params or {}
#
#         self.ne_loss_fn = NegativeEntropyRegularizer()
#
#     def _step(self, batch: HeteroData):
#         emb = self.embedding_module(batch)
#
#         q = self.clustering_module(emb)
#         ne = self.ne_loss_fn(q)
#
#         dist = -((cosine_cdist(emb, self.clustering_module.centroids) + 1) / 2)
#         assignments = dist.argmax(dim=-1)
#         counts = torch.zeros(5, dtype=torch.int32).scatter_add(0, assignments, torch.ones_like(assignments, dtype=torch.int32))
#         counts += 1
#         max_q = q.max(dim=-1).values
#         cluster_dist = torch.zeros(5, dtype=torch.float32).scatter_add(0, assignments, max_q)
#         S = cluster_dist / counts.float()
#
#         loss = torch.sum(S) + ne
#         return {
#             'loss': loss,
#         }
#
#     def configure_metrics(self):
#         return {
#             'loss': (torchmetrics.MeanMetric(), True),
#         }
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.params.get('lr', 0.001))
#         return optimizer


model = SelfSupervisedClusteringNet(
    pretrain_model, clustering_module
)
# model = FullClusteringNet(
#     embedding_module, clustering_module
# )

trainer = pl.Trainer(
    max_epochs=20, default_root_dir=str(save_path))
# trainer.fit(model, data_module.nodes_dataloader())
trainer.fit(model, data_module)


predictions = trainer.predict(model, data_module)
embeddings, assignments = map(lambda x: torch.cat(x, dim=0).detach().cpu(), zip(*predictions))
assignments = torch.argmax(assignments, dim=1)

save_path = BENCHMARKS_RESULTS.joinpath('analysis', 'pyg-sage-comopt')
save_path.mkdir(parents=True, exist_ok=True)

labeling = pd.Series(assignments.squeeze(), index=dataset.node_mapping(), name="cid")
labeling.index.name = "nid"
comlist = CommunityAssignment(labeling)
comlist.save_comlist(save_path.joinpath('schema.comlist'))

from datasets.scripts import export_to_visualization
from shared.graph import DataGraph
from shared.schema import DatasetSchema, GraphSchema
from benchmarks.evaluation import get_metric_list

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
G.to_undirected()

metrics = get_metric_list(ground_truth=False, overlapping=False)

results = pd.DataFrame([
    {
        'metric': metric_cls.metric_name(),
        'value': metric_cls.calculate(G, comlist)
    }
    for metric_cls in metrics]
)
print(results)
