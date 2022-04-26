from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from datasets.transforms.define_snapshots import DefineSnapshots
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.samplers.ballroom_sampler import BallroomSamplerParams, BallroomSampler
from ml.data.samplers.base import Sampler
from ml.data.samplers.hgt_sampler import HGTSamplerParams, HGTSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.samplers.sage_sampler import SAGESamplerParams, SAGESampler
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.layers.conv.hgt_cov_net import HGTConvNet
from ml.layers.conv.hybrid_conv_net import HybridConvNet
from ml.layers.conv.sage_conv_net import SAGEConvNet
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.het2vec import Het2VecModel, Het2VecDataModule
from ml.utils import HParams, DataLoaderParams, Metric, OptimizerParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class ConvMethod(Enum):
    SAGE = 'sage'
    HGT = 'hgt'
    NONE = 'none'


@dataclass
class MGCOMFeatModelParams(HParams):
    embed_node_types: List[NodeType] = field(default_factory=list)
    """List of node types to embed instead of using features for."""
    metric: Metric = Metric.DOTP
    """Metric to use for distance/similarity calculation."""

    repr_dim: int = 32

    conv_method: ConvMethod = ConvMethod.HGT
    conv_hidden_dim: Optional[int] = None
    conv_num_layers: int = 2
    conv_num_heads: int = 2


class MGCOMFeatModel(Het2VecModel):
    hparams: Union[MGCOMFeatModelParams, OptimizerParams]

    def __init__(
            self,
            metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
            hparams: MGCOMFeatModelParams,
            optimizer_params: Optional[OptimizerParams] = None,
    ) -> None:
        self.save_hyperparameters(hparams.to_dict())

        for node_type in self.hparams.embed_node_types:
            if node_type not in metadata[0]:
                raise ValueError(f'Node type {node_type} not in metadata')

        if self.hparams.conv_method == ConvMethod.SAGE:
            conv = SAGEConvNet(
                metadata, self.hparams.repr_dim,
                hidden_dim=self.hparams.conv_hidden_dim,
                num_layers=self.hparams.conv_num_layers,
            )
        elif self.hparams.conv_method == ConvMethod.HGT:
            conv = HGTConvNet(
                metadata, self.hparams.repr_dim,
                hidden_dim=self.hparams.conv_hidden_dim,
                num_layers=self.hparams.conv_num_layers,
                heads=self.hparams.conv_num_heads,
            )
        elif self.hparams.conv_method == ConvMethod.NONE:
            conv = None
        else:
            raise ValueError(f'Unknown conv method {self.hparams.conv_method}')

        embedder = HybridConvNet(
            metadata,
            embed_num_nodes={
                node_type: num_nodes
                for node_type, num_nodes in num_nodes_dict.items() if node_type in self.hparams.embed_node_types
            },
            conv=conv,
        )

        super().__init__(embedder, self.hparams.metric, optimizer_params)


@dataclass
class MGCOMFeatDataModuleParams(GraphDataModuleParams):
    sampler_method: ConvMethod = ConvMethod.HGT
    num_samples: List[int] = field(default_factory=lambda: [3, 2])
    """The number of nodes to sample in each iteration and for each (node type in case of HGT, and edge_type in case 
    of SAGE). """


class MGCOMFeatDataModule(Het2VecDataModule):
    hparams: Union[MGCOMFeatDataModuleParams, DataLoaderParams]

    @abstractmethod
    def _build_n2v_sampler(self, data: HeteroData, transform_meta=None) -> Union[Node2VecSampler, BallroomSampler]:
        raise NotImplementedError

    def _build_conv_sampler(self, data: HeteroData) -> Union[HGTSampler, SAGESampler]:
        if self.hparams.sampler_method == ConvMethod.HGT:
            sampler = HGTSampler(data, hparams=HGTSamplerParams(
                num_samples=self.hparams.num_samples,
            ))
        elif self.hparams.sampler_method == ConvMethod.SAGE:
            sampler = SAGESampler(data, hparams=SAGESamplerParams(
                num_samples=self.hparams.num_samples,
            ))
        else:
            raise ValueError("No sampler params provided")

        return sampler

    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        mapper = ToHeteroMappingTransform(data.num_nodes_dict)
        hgt_sampler = self._build_conv_sampler(data)

        def transform_meta(node_idx):
            node_idx_dict, node_perm_dict = mapper.transform(node_idx)
            node_meta = hgt_sampler(node_idx_dict)
            return node_meta, node_perm_dict

        n2v_sampler = self._build_n2v_sampler(data, transform_meta)
        return n2v_sampler

    def eval_sampler(self, data: HeteroData) -> Optional[Sampler]:
        return self._build_conv_sampler(data)

    @property
    def train_on_full_data(self):
        return False


@dataclass
class MGCOMTopoDataModuleParams(MGCOMFeatDataModuleParams):
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class MGCOMTopoDataModule(MGCOMFeatDataModule):
    hparams: Union[MGCOMTopoDataModuleParams, DataLoaderParams]

    def _build_n2v_sampler(self, data: HeteroData, transform_meta=None) -> Union[Node2VecSampler, BallroomSampler]:
        hdata = self.train_data.to_homogeneous(
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        n2v_sampler = Node2VecSampler(
            hdata.edge_index, hdata.num_nodes,
            hparams=self.hparams.n2v_params, transform_meta=transform_meta,
        )
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

    def _build_n2v_sampler(self, data: HeteroData, transform_meta=None) -> Union[Node2VecSampler, BallroomSampler]:
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
            hparams=self.hparams.ballroom_params,
            transform_meta=transform_meta
        )
        return ballroom_sampler
