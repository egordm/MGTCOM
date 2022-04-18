from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from ml.data.loaders.nodes_loader import NodesLoader
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams, BallroomSampler
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.layers.fc_net import FCNet, FCNetParams
from ml.models.base.embedding import BaseEmbeddingModel, EmbeddingCombineMode
from ml.models.mgcom_feat import MGCOMFeatDataModuleParams, MGCOMFeatModel, MGCOMTempoDataModule
from ml.utils import HParams, DataLoaderParams, Metric, OptimizerParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class MGCOMCombiModelParams(HParams):
    embed_node_types: List[NodeType] = field(default_factory=list)
    metric: Metric = Metric.L2

    feat_dim: int = 32
    conv_hidden_dim: Optional[int] = None
    conv_num_layers: int = 2
    conv_num_heads: int = 2

    tempo_repr_dim: int = 32
    tempo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    tempo_weight: float = 1.0

    topo_repr_dim: int = 32
    topo_hidden_dim: List[int] = field(default_factory=lambda: [32])
    topo_weight: float = 1.0

    emb_combine_mode: EmbeddingCombineMode = EmbeddingCombineMode.CONCAT


class MGCOMCombiModel(BaseEmbeddingModel):
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

        if self.hparams.emb_combine_mode != EmbeddingCombineMode.CONCAT:
            assert self.hparams.tempo_repr_dim == self.hparams.topo_repr_dim, \
                f"Temporal and topological representations must have the same dimensionality if " \
                f"combine mode is not CONCAT, but got {self.hparams.tempo_repr_dim} and " \
                f"{self.hparams.topo_repr_dim}"

        self.mgcom_feat = MGCOMFeatModel(
            metadata, num_nodes_dict,
            hparams=hparams,
            optimizer_params=None,
            add_out_net=False,
        )

        self.out_net_tempo = FCNet(
            self.hparams.feat_dim,
            hparams=FCNetParams(
                repr_dim=self.hparams.tempo_repr_dim,
                hidden_dim=self.hparams.tempo_hidden_dim,
            )
        )
        self.out_net_topo = FCNet(
            self.hparams.feat_dim,
            hparams=FCNetParams(
                repr_dim=self.hparams.topo_repr_dim,
                hidden_dim=self.hparams.topo_hidden_dim,
            )
        )

    @property
    def repr_dim(self):
        if self.hparams.emb_combine_mode.CONCAT:
            return self.hparams.tempo_repr_dim + self.hparams.topo_repr_dim
        else:
            return self.hparams.tempo_repr_dim

    def forward(self, batch):
        node_meta = batch
        Z_emb = self.mgcom_feat.embedder(node_meta)
        Z_feat = {
            node_type: self.embedding_combine_fn([
                self.out_net_tempo(z_emb),
                self.out_net_topo(z_emb),
            ])
            for node_type, z_emb in Z_emb.items()
        }

        return Z_feat

    def _forward_feat(self, node_meta) -> Tensor:
        Z_dict = self.mgcom_feat.embedder(node_meta)

        # Transform hetero data to homogenous data in the sampled order
        Z = torch.zeros(node_meta.batch_size, self.mgcom_feat.embedder.repr_dim)
        for store in node_meta.node_stores:
            node_type = store._key
            Z[store.batch_perm] = Z_dict[node_type]

        return Z

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            (tempo_pos_walks, tempo_neg_walks, tempo_node_meta),
            (topo_pos_walks, topo_neg_walks, topo_node_meta),
        ) = batch

        Z_feat_tempo = self._forward_feat(tempo_node_meta)
        Z_tempo = self.out_net_tempo(Z_feat_tempo)

        Z_feat_topo = self._forward_feat(topo_node_meta)
        Z_topo = self.out_net_topo(Z_feat_topo)

        loss_tempo = self.mgcom_feat.n2v_model.loss(tempo_pos_walks, tempo_neg_walks, Z_tempo)
        loss_topo = self.mgcom_feat.n2v_model.loss(topo_pos_walks, topo_neg_walks, Z_topo)
        loss = self.hparams.tempo_weight * loss_tempo + self.hparams.topo_weight * loss_topo

        return loss


@dataclass
class MGCOMCombiDataModuleParams(MGCOMFeatDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()
    window: Optional[Tuple[int, int]] = None
    ballroom_params: BallroomSamplerParams = BallroomSamplerParams()


class MGCOMCombiDataModule(MGCOMTempoDataModule):
    hparams: Union[MGCOMCombiDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMCombiDataModuleParams,
            loader_params: DataLoaderParams
    ) -> None:
        super().__init__(dataset, hparams, loader_params)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        hdata = to_homogeneous(
            self.train_data,
            node_attrs=['timestamp_from'], edge_attrs=['timestamp_from'],
            add_node_type=False, add_edge_type=False
        )
        ballroom_sampler = BallroomSampler(
            hdata.node_timestamp_from,
            hdata.edge_index,
            hdata.edge_timestamp_from,
            tuple(self.hparams.window),
            hparams=self.hparams.ballroom_params
        )
        n2v_sampler = Node2VecSampler(hdata.edge_index, hdata.num_nodes, hparams=self.hparams.n2v_params)
        hgt_sampler = self._build_hgt_sampler(self.train_data)

        sampler_tempo = HybridSampler(
            n2v_sampler=ballroom_sampler,
            hgt_sampler=hgt_sampler,
        )
        sampler_topo = HybridSampler(
            n2v_sampler=n2v_sampler,
            hgt_sampler=hgt_sampler,
        )

        def custom_transform(node_ids):
            return (
                sampler_tempo(node_ids),
                sampler_topo(node_ids),
            )

        return NodesLoader(
            self.train_data.num_nodes, transform=custom_transform,
            shuffle=True,
            **self.loader_params.to_dict(),
        )
