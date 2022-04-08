from typing import Optional, Any

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from datasets.utils.conversion import igraph_from_hetero
from ml.layers.clustering import KMeans
from ml.layers.metrics.modularity import newman_girvan_modularity
from ml.utils.dict import merge_dicts


class ModularityMonitorCallback(Callback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        super().setup(trainer, pl_module, stage)
        self.data = pl_module.dataset.data
        G, _, _, _ = igraph_from_hetero(self.data, node_attrs=dict(label=self.data.name_dict))
        comm = G.community_multilevel()

        self.k = len(comm)
        self.clus = KMeans(pl_module.hparams.repr_dim, self.k, sim=pl_module.hparams.sim)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        emb_dict = pl_module.val_emb_dict

        emb_flat = torch.cat(list(emb_dict.values()), dim=0)
        if len(emb_flat) != self.data.num_nodes:
            return

        self.clus.fit(emb_flat)

        self.I = {
            node_type: self.clus.assign(emb)
            for node_type, emb in emb_dict.items()
        }
        self.m = newman_girvan_modularity(self.data, self.I, self.k)

        for logger in trainer.loggers:
            logger.log_metrics({"val_modularity": self.m}, step=trainer.global_step)

        pl_module.log("epoch_modularity", self.m, prog_bar=True)
