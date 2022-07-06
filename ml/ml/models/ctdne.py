from dataclasses import dataclass
from typing import Union, Optional

from torch_geometric.data import Data, HeteroData

from datasets import GraphDataset
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.data.samplers.base import Sampler
from ml.data.samplers.ctdne_sampler import CTDNESampler, CTDNESamplerParams
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.base.hgraph_datamodule import HomogenousGraphDataModule
from ml.models.node2vec import Node2VecModel, Node2VecClusModel
from ml.utils import DataLoaderParams


class CTDNEModel(Node2VecClusModel):
    pass


@dataclass
class CTDNEDataModuleParams(GraphDataModuleParams):
    ctdne_params: CTDNESamplerParams = CTDNESamplerParams()


class CTDNEDataModule(HomogenousGraphDataModule):
    hparams: Union[CTDNEDataModuleParams, DataLoaderParams]

    def train_sampler(self, data: Data) -> Optional[Sampler]:
        return CTDNESampler(
            data.node_timestamp_from,
            data.edge_index,
            data.edge_timestamp_from,
            hparams=self.hparams.ctdne_params
        )

    def eval_sampler(self, data: Data) -> Optional[Sampler]:
        return None

    def to_homogenous(self, data: HeteroData) -> Data:
        if isinstance(data, Data):
            return data
        else:
            return to_homogeneous(
                data,
                node_attrs=['timestamp_from', 'train_mask', 'val_mask', 'test_mask'],
                edge_attrs=['timestamp_from', 'train_mask', 'val_mask', 'test_mask'],
            )
