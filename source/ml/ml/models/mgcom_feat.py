from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from datasets.transforms.define_snapshots import DefineSnapshots
from datasets.utils.base import Snapshots
from ml.data.graph_datamodule import GraphDataModule, GraphDataModuleParams
from ml.data.loaders.nodes_loader import NodesLoader, HeteroNodesLoader
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams, BallroomSampler
from ml.data.samplers.hgt_sampler import HGTSamplerParams, HGTSampler
from ml.data.samplers.hybrid_sampler import HybridSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.transforms.ensure_timestamps import EnsureTimestampsTransform
from ml.data.transforms.eval_split import EvalNodeSplitTransform
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs
from ml.layers.fc_net import FCNet, FCNetParams
from ml.layers.hybrid_conv_net import HybridConvNet, HybridConvNetParams
from ml.models.base.embedding import BaseEmbeddingModel
from ml.models.node2vec import Node2VecModel
from ml.utils import HParams, DataLoaderParams, Metric, OptimizerParams
from ml.utils.labelling import NodeLabelling, extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class MGCOMFeatModelParams(HParams):
    embed_node_types: List[NodeType] = field(default_factory=list)
    metric: Metric = Metric.L2

    feat_dim: int = 32
    conv_hidden_dim: Optional[int] = None
    conv_num_layers: int = 2
    conv_num_heads: int = 2

    repr_dim: int = 32
    hidden_dim: List[int] = field(default_factory=lambda: [32])


class MGCOMFeatModel(BaseEmbeddingModel):
    hparams: Union[MGCOMFeatModelParams, OptimizerParams]

    def __init__(
            self,
            metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
            hparams: MGCOMFeatModelParams,
            optimizer_params: Optional[OptimizerParams] = None,
            add_out_net: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        if optimizer_params is not None:
            self.save_hyperparameters(optimizer_params.to_dict())

        self.embedder = HybridConvNet(
            metadata,
            embed_num_nodes={
                node_type: num_nodes
                for node_type, num_nodes in num_nodes_dict.items() if node_type in self.hparams.embed_node_types
            },
            hparams=HybridConvNetParams(
                repr_dim=self.hparams.feat_dim,
                hidden_dim=self.hparams.conv_hidden_dim,
                num_layers=self.hparams.conv_num_layers,
                num_heads=self.hparams.conv_num_heads,
            )
        )
        if add_out_net:
            self.out_net = FCNet(
                self.hparams.feat_dim,
                hparams=FCNetParams(
                    repr_dim=self.hparams.repr_dim,
                    hidden_dim=self.hparams.hidden_dim,
                )
            )

        # noinspection PyTypeChecker
        self.n2v_model = Node2VecModel(
            embedder=None,
            metric=self.hparams.metric,
        )

    @property
    def repr_dim(self):
        return self.hparams.repr_dim

    def forward(self, batch):
        node_meta = batch
        Z_emb = self.embedder(node_meta)
        Z_feat = {
            node_type: self.out_net(z_emb)
            for node_type, z_emb in Z_emb.items()
        }

        return Z_feat

    def _forward_emb(self, node_meta) -> Tensor:
        Z_dict = self.embedder(node_meta)

        # Transform hetero data to homogenous data in the sampled order
        Z = torch.zeros(node_meta.batch_size, self.embedder.repr_dim)
        for store in node_meta.node_stores:
            node_type = store._key
            Z[store.batch_perm] = Z_dict[node_type]

        return Z

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pos_walks, neg_walks, node_meta = batch

        Z_emb = self._forward_emb(node_meta)
        Z_feat = self.out_net(Z_emb)

        loss = self.n2v_model.loss(pos_walks, neg_walks, Z_feat)
        return loss


@dataclass
class MGCOMFeatDataModuleParams(GraphDataModuleParams):
    hgt_params: HGTSamplerParams = HGTSamplerParams()


class MGCOMFeatDataModule(GraphDataModule):
    hparams: Union[MGCOMFeatDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMFeatDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__(dataset, hparams, loader_params)

    @abstractmethod
    def _build_n2v_sampler(self, data: HeteroData) -> Union[Node2VecSampler, BallroomSampler]:
        raise NotImplementedError

    def _build_hgt_sampler(self, data: HeteroData) -> HGTSampler:
        return HGTSampler(data, hparams=self.hparams.hgt_params)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = HybridSampler(
            n2v_sampler=self._build_n2v_sampler(self.train_data),
            hgt_sampler=self._build_hgt_sampler(self.train_data),
        )

        return NodesLoader(
            self.train_data.num_nodes, transform=sampler,
            shuffle=True,
            **self.loader_params.to_dict(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_hgt_sampler(self.val_data)

        return HeteroNodesLoader(
            self.val_data.num_nodes_dict, transform_nodes_fn=sampler,
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_hgt_sampler(self.test_data)

        return HeteroNodesLoader(
            self.test_data.num_nodes_dict, transform_nodes_fn=sampler,
            shuffle=False,
            **self.loader_params.to_dict(),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_hgt_sampler(self.data)

        return HeteroNodesLoader(
            self.data.num_nodes_dict, transform_nodes_fn=sampler,
            shuffle=False,
            **self.loader_params.to_dict(),
        )


@dataclass
class MGCOMTopoDataModuleParams(MGCOMFeatDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class MGCOMTopoDataModule(MGCOMFeatDataModule):
    hparams: Union[MGCOMTopoDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMTopoDataModuleParams,
            loader_params: DataLoaderParams
    ) -> None:
        super().__init__(dataset, hparams, loader_params)

    def _build_n2v_sampler(self, data: HeteroData) -> Union[Node2VecSampler, BallroomSampler]:
        hdata = self.train_data.to_homogeneous(node_attrs=[], edge_attrs=[], add_node_type=False, add_edge_type=False)
        n2v_sampler = Node2VecSampler(hdata.edge_index, hdata.num_nodes, hparams=self.hparams.n2v_params)
        return n2v_sampler


@dataclass
class MGCOMTempoDataModuleParams(MGCOMFeatDataModuleParams):
    window: Optional[Tuple[int, int]] = None
    ballroom_params: BallroomSamplerParams = BallroomSamplerParams()


class MGCOMTempoDataModule(MGCOMFeatDataModule):
    hparams: Union[MGCOMTempoDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: MGCOMFeatDataModuleParams,
            loader_params: DataLoaderParams
    ) -> None:
        if hparams.window is None:
            if isinstance(dataset, GraphDataset) and dataset.snapshots is not None:
                logger.warning("No temporal window specified, trying to infer it from dataset snapshots")
                snapshot_key = max(dataset.snapshots.keys())
                snapshot = dataset.snapshots[snapshot_key]
            else:
                logger.warning('Dataset does not have snapshots, trying to create snapshots from dataset')
                snapshot = DefineSnapshots(10)(dataset.data)

            hparams.window = (0, int(snapshot[0][1] - snapshot[0][0]))
            logger.warning(f"Inferred temporal window: {hparams.window}")

        super().__init__(dataset, hparams, loader_params)

    def _build_n2v_sampler(self, data: HeteroData) -> Union[Node2VecSampler, BallroomSampler]:
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
        return ballroom_sampler
