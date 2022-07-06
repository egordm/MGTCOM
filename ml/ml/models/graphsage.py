from dataclasses import dataclass, field
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Union

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.algo.transforms import ToHeteroMappingTransform
from ml.data.samplers.ballroom_sampler import BallroomSampler
from ml.data.samplers.base import Sampler
from ml.data.samplers.hgt_sampler import HGTSampler
from ml.data.samplers.node2vec_sampler import Node2VecSampler, Node2VecSamplerParams
from ml.data.samplers.sage_sampler import SAGESamplerParams, SAGESampler
from ml.layers.conv.hybrid_conv_net import HybridConvNet
from ml.layers.conv.sage_conv_net import SAGEConvNet
from ml.models.base.clustering_mixin import ClusteringMixinParams
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.het2vec import Het2VecModel, Het2VecDataModule, Het2VecClusModel, Het2VecClusModelParams
from ml.models.node2vec import Node2VecModelParams
from ml.utils import DataLoaderParams, OptimizerParams
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class GraphSAGEModelParams(Het2VecClusModelParams, ClusteringMixinParams):
    conv_hidden_dim: Optional[int] = None
    """Hidden dimension of the convolution layers. If None, use repr_dim."""
    conv_num_layers: int = 2
    """Number of convolution layers."""


class GraphSAGEModel(Het2VecClusModel):
    hparams: Union[GraphSAGEModelParams, OptimizerParams]

    def __init__(
        self,
        metadata: Metadata, num_nodes_dict: Dict[NodeType, int],
        hparams: GraphSAGEModelParams,
        optimizer_params: Optional[OptimizerParams] = None,
    ) -> None:
        self.save_hyperparameters(hparams.to_dict())

        conv = SAGEConvNet(
            metadata, self.hparams.repr_dim,
            hidden_dim=self.hparams.conv_hidden_dim,
            num_layers=self.hparams.conv_num_layers,
        )

        embedder = HybridConvNet(
            metadata,
            embed_num_nodes={},
            conv=conv,
            hidden_dim=self.hparams.conv_hidden_dim,
        )

        super().__init__(embedder, hparams, optimizer_params)


@dataclass
class GraphSAGEDataModuleParams(GraphDataModuleParams):
    num_samples: List[int] = field(default_factory=lambda: [3, 2])
    """The number of nodes to sample in each iteration and for each (node type in case of HGT, and edge_type in case 
    of SAGE). """
    n2v_params: Node2VecSamplerParams = Node2VecSamplerParams()


class GraphSAGEDataModule(Het2VecDataModule):
    hparams: Union[GraphSAGEDataModuleParams, DataLoaderParams]

    def __init__(self, dataset: GraphDataset, hparams: GraphDataModuleParams, loader_params: DataLoaderParams) -> None:
        super().__init__(dataset, hparams, loader_params)

        def homogenify(data: HeteroData):
            hdata = to_homogeneous(data,
                node_attrs=None, edge_attrs=None,
                add_node_type=False, add_edge_type=False
            )
            hdata.node_stores[0]._node_type_names = ['0']
            hdata.node_stores[0]._edge_type_names = [('0', '0', '0')]

            hetdata = hdata.to_heterogeneous()
            for store in hetdata.stores:
                keys = list(hetdata.keys)
                for key in keys:
                    if key == 'edge_index':
                        continue

                    if key == 'x':
                        continue

                    if key in store:
                        if 'edge_' in key and isinstance(store, EdgeStorage):
                            store[key.replace('edge_', '')] = store[key]
                        elif 'node_' in key and isinstance(store, NodeStorage):
                            store[key.replace('node_', '')] = store[key]

                        del store[key]

            feat_dim = max(x.shape[1] for _, x in data.x_dict.items())
            X = torch.zeros(data.num_nodes, feat_dim)
            offset = 0
            for key in data.node_types:
                x = data[key].x
                X[offset:offset+x.shape[0], :x.shape[1]] = x
                offset += x.shape[0]

            hetdata['0'].x = X

            return hetdata

        self.data, self.train_data, self.val_data, self.test_data = (
            homogenify(self.data),
            homogenify(self.train_data),
            homogenify(self.val_data),
            homogenify(self.test_data),
        )
        u = 0

    @property
    def metadata(self) -> Metadata:
        return self.data.metadata()

    def _build_conv_sampler(self, data: HeteroData) -> Union[HGTSampler, SAGESampler]:
        sampler = SAGESampler(data, hparams=SAGESamplerParams(
            num_samples=self.hparams.num_samples,
        ))

        return sampler

    def _build_n2v_sampler(self, data: HeteroData, transform_meta=None) -> Union[Node2VecSampler, BallroomSampler]:
        hdata = data.to_homogeneous(
            node_attrs=[], edge_attrs=[],
            add_node_type=False, add_edge_type=False
        )
        n2v_sampler = Node2VecSampler(
            hdata.edge_index, hdata.num_nodes,
            hparams=self.hparams.n2v_params, transform_meta=transform_meta,
        )
        return n2v_sampler

    def train_sampler(self, data: HeteroData) -> Optional[Sampler]:
        mapper = ToHeteroMappingTransform(data.num_nodes_dict)
        sage_sampler = self._build_conv_sampler(data)

        def transform_meta(node_idx):
            node_idx_dict, node_perm_dict = mapper.transform(node_idx)
            node_meta = sage_sampler(node_idx_dict)
            return node_meta, node_perm_dict

        n2v_sampler = self._build_n2v_sampler(data, transform_meta)
        return n2v_sampler

    def eval_sampler(self, data: HeteroData) -> Optional[Sampler]:
        return self._build_conv_sampler(data)
