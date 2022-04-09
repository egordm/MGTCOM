from typing import Optional

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from datasets.utils.conversion import igraph_from_hetero
from ml.layers.clustering import KMeans
from ml.layers.metrics import newman_girvan_modularity, silhouette_score, davies_bouldin_score


class ModularityMonitorCallback(Callback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        super().setup(trainer, pl_module, stage)
        self.data = pl_module.dataset.data
        self.G, _, _, self.node_offsets = igraph_from_hetero(self.data, node_attrs=dict(label=self.data.name_dict))
        comm = self.G.community_multilevel()

        self.k = len(comm)
        self.sim = pl_module.hparams.sim
        self.clus = KMeans(pl_module.hparams.repr_dim, self.k, sim=self.sim)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        emb_dict = pl_module.val_emb_dict

        emb_flat = torch.zeros((self.G.vcount(), pl_module.hparams.repr_dim))
        for node_type, offset in self.node_offsets.items():
            emb_flat[offset:offset + len(emb_dict[node_type])] = emb_dict[node_type]

        self.clus.fit(emb_flat)
        I_flat = self.clus.assign(emb_flat)
        I = {
            node_type: I_flat[offset:offset + len(emb_dict[node_type])]
            for node_type, offset in self.node_offsets.items()
        }

        modularity = newman_girvan_modularity(self.data, I, self.k)
        sc = silhouette_score(emb_flat.numpy(), I_flat.numpy(), sim=self.sim)
        db = davies_bouldin_score(emb_flat.numpy(), I_flat.numpy(), sim=self.sim)

        pl_module.log_dict({
            "epoch_m": modularity,
            "epoch_sc": sc,
            "epoch_dbs": db,
        }, prog_bar=True)

        for logger in trainer.loggers:
            logger.log_metrics({
                "modularity": modularity,
                "silhouette_score": sc,
                "davies_bouldin_score": db
            }, step=trainer.global_step)
