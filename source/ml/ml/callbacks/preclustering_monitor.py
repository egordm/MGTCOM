from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from datasets.utils.conversion import igraph_from_hetero
from ml.layers.clustering import KMeans
from ml.layers.metrics import newman_girvan_modularity, silhouette_score, davies_bouldin_score
from ml.models.base.embedding import BaseEmbeddingModel
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class PreclusteringMonitor(Callback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        super().setup(trainer, pl_module, stage)
        self.data = pl_module.dataset.data
        self.G, _, _, self.node_offsets = igraph_from_hetero(self.data, node_attrs=dict(label=self.data.name_dict))
        comm = self.G.community_multilevel()

        self.k = len(comm)
        self.sim = pl_module.hparams.sim
        self.clus = KMeans(pl_module.hparams.repr_dim, self.k, sim=self.sim)
        self.repr_dim = pl_module.hparams.repr_dim

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseEmbeddingModel) -> None:
        if trainer.state.stage != RunningStage.VALIDATING:
            return

        logger.info(f"Running preclustering")
        emb_dict = pl_module.val_embs
        emb_flat, I_flat = self.run_preclutering(emb_dict)
        I = {
            node_type: I_flat[offset:offset + len(emb_dict[node_type])]
            for node_type, offset in self.node_offsets.items()
        }

        modularity = newman_girvan_modularity(self.data, I, self.k)
        sc = silhouette_score(emb_flat, I_flat, sim=self.sim)
        db = davies_bouldin_score(emb_flat, I_flat, sim=self.sim)

        pl_module.log_dict({
            "epoch_m": modularity,
            "epoch_sc": sc,
            "epoch_dbs": db,
        }, prog_bar=True, logger=True)

    def run_preclutering(self, emb_dict: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        emb_flat = torch.zeros((self.G.vcount(), self.repr_dim))
        for node_type, offset in self.node_offsets.items():
            emb_flat[offset:offset + len(emb_dict[node_type])] = emb_dict[node_type]

        self.clus.fit(emb_flat)
        I_flat = self.clus.assign(emb_flat)

        return emb_flat, I_flat
