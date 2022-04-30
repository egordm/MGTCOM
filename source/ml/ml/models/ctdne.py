from dataclasses import dataclass
from typing import Union, Optional

from torch_geometric.data import Data

from datasets import GraphDataset
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.data.samplers.base import Sampler
from ml.data.samplers.ctdne_sampler import CTDNESampler, CTDNESamplerParams
from ml.models.base.graph_datamodule import GraphDataModuleParams
from ml.models.base.hgraph_datamodule import HomogenousGraphDataModule
from ml.models.node2vec import Node2VecModel
from ml.utils import DataLoaderParams


class CTDNEModel(Node2VecModel):
    pass


@dataclass
class CTDNEDataModuleParams(GraphDataModuleParams):
    ctdne_params: CTDNESamplerParams = CTDNESamplerParams()


class CTDNEDataModule(HomogenousGraphDataModule):
    hparams: Union[CTDNEDataModuleParams, DataLoaderParams]

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: CTDNEDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__(dataset, hparams, loader_params)
        hdata = to_homogeneous(
            self.data,
            node_attrs=['timestamp_from', 'train_mask', 'val_mask', 'test_mask'],
            edge_attrs=['timestamp_from', 'train_mask', 'val_mask', 'test_mask'],
        )
        self.train_data, self.val_data, self.test_data = hdata, hdata, hdata  # Since induction doesnt work on node2vec

    def train_sampler(self, data: Data) -> Optional[Sampler]:
        return CTDNESampler(
            self.train_data.node_timestamp_from,
            self.train_data.edge_index,
            self.train_data.edge_timestamp_from,
            hparams=self.hparams.ctdne_params
        )

    def eval_sampler(self, data: Data) -> Optional[Sampler]:
        return None
