from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from simple_parsing import field
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams
from ml.data.samplers.base import Sampler
from ml.data.samplers.node2vec_sampler import Node2VecSamplerParams
from ml.layers.fc_net import FCNet, FCNetParams
from ml.models.base.feature_model import FeatureCombineMode, HeteroFeatureModel
from ml.models.mgcom_feat import MGCOMFeatDataModuleParams, MGCOMFeatModel, MGCOMTempoDataModule, MGCOMTopoDataModule, \
    MGCOMFeatModelParams
from ml.utils import DataLoaderParams, OptimizerParams, dict_mapv
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class MGCOMCombiModelParams(MGCOMFeatModelParams):
    use_topo: bool = True
    use_tempo: bool = True

    topo_repr_dim: int = 32
    """Dimension of the topology feature representation vectors."""
    topo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    topo_weight: float = 1.0

    tempo_repr_dim: int = 32
    """Dimension of the temporal feature representation vectors."""
    tempo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    tempo_weight: float = 1.0

    emb_combine_mode: FeatureCombineMode = FeatureCombineMode.CONCAT


class MGCOMCombiModel(HeteroFeatureModel):
    hparams: Union[MGCOMCombiModelParams, OptimizerParams]
    pretraining: bool = True

    def __init__(
        self,
        metadata: Metadata,
        num_nodes_dict: Dict[NodeType, int],
        hparams: MGCOMCombiModelParams,
        optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(optimizer_params)
        self.save_hyperparameters(hparams.to_dict())
        self.embedding_combine_fn = self.hparams.emb_combine_mode.combine_fn

        if self.hparams.emb_combine_mode != FeatureCombineMode.CONCAT:
            assert self.hparams.topo_repr_dim == self.hparams.tempo_repr_dim, \
                f"Temporal and topological representations must have the same dimensionality if " \
                f"combine mode is not CONCAT, but got {self.hparams.tempo_repr_dim} and " \
                f"{self.hparams.topo_repr_dim}"

        assert self.hparams.use_topo or self.hparams.use_tempo, \
            f"At least one of use_topo and use_tempo must be True."

        self.repr_dim_ = self.compute_repr_dim(self.hparams)

        self.feat_net = MGCOMFeatModel(
            metadata, num_nodes_dict,
            hparams=hparams,
            optimizer_params=None,
        )

        if self.hparams.use_topo:
            self.topo_net = FCNet(
                self.hparams.repr_dim,
                hparams=FCNetParams(
                    repr_dim=self.hparams.topo_repr_dim,
                    hidden_dim=self.hparams.topo_hidden_dim,
                )
            )

        if self.hparams.use_tempo:
            self.tempo_net = FCNet(
                self.hparams.repr_dim,
                hparams=FCNetParams(
                    repr_dim=self.hparams.tempo_repr_dim,
                    hidden_dim=self.hparams.tempo_hidden_dim,
                )
            )

        # Combine the embeddings
        if self.hparams.use_topo and self.hparams.use_tempo:
            self.combi_feat_net = lambda z_emb: self.embedding_combine_fn([
                self.tempo_net(z_emb),
                self.topo_net(z_emb),
            ])
        elif self.hparams.use_topo:
            self.combi_feat_net = self.topo_net
        elif self.hparams.use_tempo:
            self.combi_feat_net = self.tempo_net

    @property
    def repr_dim(self):
        return self.repr_dim_

    @staticmethod
    def compute_repr_dim(hparams: MGCOMCombiModelParams) -> int:
        if hparams.emb_combine_mode.CONCAT:
            return (
                hparams.topo_repr_dim * int(hparams.use_topo)
                + hparams.topo_repr_dim * int(hparams.use_tempo)
            )
        else:
            return hparams.topo_repr_dim if hparams.use_topo else hparams.tempo_repr_dim

    def forward(self, batch):
        Z_emb = self.feat_net(batch)
        Z_feat = dict_mapv(Z_emb, self.combi_feat_net)
        return Z_feat

    def training_step_topo(self, batch):
        if not self.hparams.use_topo:
            return None, None

        (pos_walks, neg_walks, node_meta) = batch

        Z_emb = self.feat_net.forward_emb_flat(node_meta)
        Z = self.topo_net(Z_emb)
        loss = self.feat_net.n2v.loss(pos_walks, neg_walks, Z)

        return loss, Z

    def training_step_tempo(self, batch):
        if not self.hparams.use_tempo:
            return None, None

        (pos_walks, neg_walks, node_meta) = batch

        Z_emb = self.feat_net.forward_emb_flat(node_meta)
        Z = self.tempo_net(Z_emb)
        loss = self.feat_net.n2v.loss(pos_walks, neg_walks, Z)

        return loss, Z

    def training_step(self, batch, batch_idx, r=None) -> STEP_OUTPUT:
        (topo_walks, tempo_walks) = batch

        loss_topo, Z_topo = self.training_step_topo(topo_walks)
        loss_tempo, Z_tempo = self.training_step_tempo(tempo_walks)

        loss, out = 0.0, {}
        if loss_topo is not None:
            loss += self.hparams.topo_weight * loss_topo
            out['loss_topo'] = loss_topo.detach()
        if loss_tempo is not None:
            loss += self.hparams.tempo_weight * loss_tempo
            out['loss_tempo'] = loss_tempo.detach()

        if self.use_cluster_loss and self.cluster_module.k > 1 and r is not None:
            if self.hparams.use_topo and self.hparams.use_tempo:
                Z_combi = self.embedding_combine_fn([Z_topo, Z_tempo])
                idx = topo_pos_walks[:, 0]
            else:
                Z_combi = Z_topo if self.hparams.use_topo else Z_tempo
                idx = topo_pos_walks[:, 0] if self.hparams.use_topo else tempo_pos_walks[:, 0]

            Z_combi_i = Z_combi[idx, :]
            r_i = r[idx, :]
            mus = self.cluster_module.mus
            loss_cluster = self.cluster_loss_fn(Z_combi_i, r_i, mus)
            loss += self.hparams.cluster_weight * loss_cluster
            # loss = self.hparams.cluster_weight * loss_cluster
            out["loss_cluster"] = loss_cluster.detach()

        return {
            "loss": loss,
            **out,
        }

    @property
    def use_cluster_loss(self):
        return self.hparams.use_cluster and not self.pretraining

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        super().training_epoch_end(outputs)

        if self.hparams.use_topo:
            self.log('epoch_loss_topo', self.train_outputs.extract_mean('loss_topo'), prog_bar=True)
        if self.hparams.use_tempo:
            self.log('epoch_loss_tempo', self.train_outputs.extract_mean('loss_tempo'), prog_bar=True)
        if self.use_cluster_loss:
            self.log('epoch_loss_cluster', self.train_outputs.extract_mean('loss_cluster'), prog_bar=True)


@dataclass
class MGCOMCombiDataModuleParams(MGCOMFeatDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()
    window: Optional[Tuple[int, int]] = None
    ballroom_params: BallroomSamplerParams = BallroomSamplerParams()
    use_tempo_loader: bool = field(default=True, cmd=False)
    use_topo_loader: bool = field(default=True, cmd=False)


class MGCOMCombiDataModule(MGCOMTempoDataModule, MGCOMTopoDataModule):
    hparams: Union[MGCOMCombiDataModuleParams, DataLoaderParams]

    def __init__(
        self,
        dataset: GraphDataset,
        hparams: MGCOMCombiDataModuleParams,
        loader_params: DataLoaderParams
    ) -> None:
        if hparams.use_tempo_loader:
            MGCOMTempoDataModule.__init__(self, dataset, hparams, loader_params)
        else:
            MGCOMTopoDataModule.__init__(self, dataset, hparams, loader_params)

    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        mapper = ToHeteroMappingTransform(data.num_nodes_dict)
        hgt_sampler = self._build_conv_sampler(data)

        def transform_meta(node_idx):
            node_idx_dict, node_perm_dict = mapper.transform(node_idx)
            node_meta = hgt_sampler(node_idx_dict)
            return node_meta, node_perm_dict

        n2v_sampler = MGCOMTopoDataModule._build_n2v_sampler(self, data, transform_meta) \
            if self.hparams.use_topo_loader else None
        ballroom_sampler = MGCOMTempoDataModule._build_n2v_sampler(self, data, transform_meta) \
            if self.hparams.use_tempo_loader else None

        def combined_sampler(node_ids):
            return (
                n2v_sampler(node_ids) if self.hparams.use_topo_loader else (None, None, None),
                ballroom_sampler(node_ids) if self.hparams.use_tempo_loader else (None, None, None),
            )

        # noinspection PyTypeChecker
        return combined_sampler
