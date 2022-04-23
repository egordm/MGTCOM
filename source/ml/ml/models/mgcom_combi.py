from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams, BallroomSampler
from ml.data.samplers.base import Sampler
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.layers.fc_net import FCNet, FCNetParams
from ml.models.base.embedding import HeteroFeatureModel, FeatureCombineMode
from ml.models.mgcom_feat import MGCOMFeatDataModuleParams, MGCOMFeatModel, MGCOMTempoDataModule, MGCOMTopoDataModule, \
    MGCOMFeatModelParams
from ml.utils import HParams, DataLoaderParams, Metric, OptimizerParams, dict_mapv
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class MGCOMCombiModelParams(MGCOMFeatModelParams):
    topo_repr_dim: int = 32
    topo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    topo_weight: float = 1.0

    tempo_repr_dim: int = 32
    tempo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    tempo_weight: float = 1.0

    emb_combine_mode: FeatureCombineMode = FeatureCombineMode.CONCAT


class MGCOMCombiModel(HeteroFeatureModel):
    hparams: Union[MGCOMCombiModelParams, OptimizerParams]

    def __init__(
            self,
            metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
            hparams: MGCOMCombiModelParams,
            optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.save_hyperparameters(optimizer_params.to_dict())
        self.embedding_combine_fn = self.hparams.emb_combine_mode.combine_fn

        if self.hparams.emb_combine_mode != FeatureCombineMode.CONCAT:
            assert self.hparams.topo_repr_dim == self.hparams.tempo_repr_dim, \
                f"Temporal and topological representations must have the same dimensionality if " \
                f"combine mode is not CONCAT, but got {self.hparams.tempo_repr_dim} and " \
                f"{self.hparams.topo_repr_dim}"

        self.feat_net = MGCOMFeatModel(
            metadata, num_nodes_dict,
            hparams=hparams,
            optimizer_params=None,
        )

        self.topo_net = FCNet(
            self.hparams.repr_dim,
            hparams=FCNetParams(
                repr_dim=self.hparams.topo_repr_dim,
                hidden_dim=self.hparams.topo_hidden_dim,
            )
        )
        self.tempo_net = FCNet(
            self.hparams.repr_dim,
            hparams=FCNetParams(
                repr_dim=self.hparams.tempo_repr_dim,
                hidden_dim=self.hparams.tempo_hidden_dim,
            )
        )

    @property
    def repr_dim(self):
        if self.hparams.emb_combine_mode.CONCAT:
            return self.hparams.tempo_repr_dim + self.hparams.topo_repr_dim
        else:
            return self.hparams.tempo_repr_dim

    def forward(self, batch):
        Z_emb = self.feat_net(batch)
        Z_feat = dict_mapv(
            Z_emb,
            lambda z_emb: self.embedding_combine_fn([
                self.tempo_net(z_emb),
                self.topo_net(z_emb),
            ])
        )
        return Z_feat

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            (topo_pos_walks, topo_neg_walks, topo_node_meta),
            (tempo_pos_walks, tempo_neg_walks, tempo_node_meta),
        ) = batch

        Z_emb_topo = self.feat_net.forward_emb_flat(topo_node_meta)
        Z_emb_tempo = self.feat_net.forward_emb_flat(tempo_node_meta)

        Z_topo = self.topo_net(Z_emb_topo)
        Z_tempo = self.tempo_net(Z_emb_tempo)

        loss_topo = self.feat_net.n2v.loss(topo_pos_walks, topo_neg_walks, Z_topo)
        loss_tempo = self.feat_net.n2v.loss(tempo_pos_walks, tempo_neg_walks, Z_tempo)
        loss = self.hparams.topo_weight * loss_topo + self.hparams.tempo_weight * loss_tempo

        return loss


@dataclass
class MGCOMCombiDataModuleParams(MGCOMFeatDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()
    window: Optional[Tuple[int, int]] = None
    ballroom_params: BallroomSamplerParams = BallroomSamplerParams()


class MGCOMCombiDataModule(MGCOMTempoDataModule, MGCOMTopoDataModule):
    hparams: Union[MGCOMCombiDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMCombiDataModuleParams,
            loader_params: DataLoaderParams
    ) -> None:
        MGCOMTempoDataModule.__init__(self, dataset, hparams, loader_params)

    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        mapper = ToHeteroMappingTransform(data.num_nodes_dict)
        hgt_sampler = self._build_conv_sampler(data)

        def transform_meta(node_idx):
            node_idx_dict, node_perm_dict = mapper.transform(node_idx)
            node_meta = hgt_sampler(node_idx_dict)
            return node_meta, node_perm_dict

        n2v_sampler = MGCOMTopoDataModule._build_n2v_sampler(self, data, transform_meta)
        ballroom_sampler = MGCOMTempoDataModule._build_n2v_sampler(self, data, transform_meta)

        def combined_sampler(node_ids):
            return (
                n2v_sampler(node_ids),
                ballroom_sampler(node_ids),
            )

        # noinspection PyTypeChecker
        return combined_sampler
