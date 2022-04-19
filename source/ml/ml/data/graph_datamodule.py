from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Union, Tuple

import pytorch_lightning as pl
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from datasets.utils.base import Snapshots
from ml.data.transforms.ensure_timestamps import EnsureTimestampsTransform
from ml.data.transforms.eval_split import EvalNodeSplitTransform
from ml.data.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs
from ml.utils import HParams, DataLoaderParams
from ml.utils.labelling import NodeLabelling, extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from shared import get_logger

logger = get_logger(Path(__file__).stem)


@dataclass
class GraphDataModuleParams(HParams):
    lp_max_pairs: int = 5000


class GraphDataModule(pl.LightningDataModule):
    dataset: GraphDataset
    hparams: Union[GraphDataModuleParams, DataLoaderParams]
    loader_params: DataLoaderParams

    def __init__(
            self,
            dataset: GraphDataset,
            hparams: GraphDataModuleParams,
            loader_params: DataLoaderParams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams.to_dict())
        self.save_hyperparameters(loader_params.to_dict())
        self.loader_params = loader_params

        self.dataset = dataset
        self.data = EnsureTimestampsTransform(warn=True)(dataset.data)
        self.train_data, self.val_data, self.test_data = EvalNodeSplitTransform()(self.data)

    @property
    def metadata(self) -> Metadata:
        return self.dataset.metadata

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        return self.train_data.num_nodes_dict

    def _edge_prediction_pairs(self, data: HeteroData, mask_name: str = 'train_mask') -> Tuple[Tensor, Tensor]:
        """
        It takes a heterogeneous graph and returns a tuple of two tensors, the edges and the edge labels.

        :param data: HeteroData
        :type data: HeteroData
        :param mask_name: The name of the edge mask attribute, defaults to train_mask
        :type mask_name: str (optional)
        """
        hdata = to_homogeneous(
            data,
            node_attrs=[], edge_attrs=[mask_name],
            add_node_type=False, add_edge_type=False
        )
        return extract_edge_prediction_pairs(
            hdata.edge_index, hdata.num_nodes, getattr(hdata, f'edge_{mask_name}'),
            max_samples=self.hparams.lp_max_pairs
        )

    def train_prediction_pairs(self) -> Tuple[Tensor, Tensor]:
        return self._edge_prediction_pairs(self.train_data, 'train_mask')

    def val_prediction_pairs(self) -> Tuple[Tensor, Tensor]:
        return self._edge_prediction_pairs(self.val_data, 'val_mask')

    def test_prediction_pairs(self) -> Tuple[Tensor, Tensor]:
        return self._edge_prediction_pairs(self.test_data, 'test_mask')

    def _extract_inferred_labels(
            self, data: HeteroData, snapshots: Optional[Dict[int, Snapshots]] = None
    ) -> Dict[str, NodeLabelling]:
        """
        It extracts the Louvain labels, and if snapshots are provided, it extracts the snapshot labels

        :param data: The data object that contains the graph and node features
        :type data: HeteroData
        :param snapshots: Optional[Dict[int, Snapshots]] = None
        :type snapshots: Optional[Dict[int, Snapshots]]
        :return: A dictionary of node labels.
        """
        logger.info('Extracting labels for visualization')
        node_labels = {}
        node_labels['Louvain Labels'] = extract_louvain_labels(data)

        if snapshots:
            node_timestamps = extract_timestamp_labels(data)
            for i, snapshot in snapshots.items():
                snapshot_labels = extract_snapshot_labels(node_timestamps, snapshot)
                node_labels[f'{i} Temporal Snapshots'] = snapshot_labels

        return node_labels

    @lru_cache(maxsize=1)
    def train_inferred_labels(self) -> Dict[str, NodeLabelling]:
        return self._extract_inferred_labels(self.train_data)

    @lru_cache(maxsize=1)
    def val_inferred_labels(self) -> Dict[str, NodeLabelling]:
        return self._extract_inferred_labels(self.val_data)

    @lru_cache(maxsize=1)
    def test_inferred_labels(self) -> Dict[str, NodeLabelling]:
        return self._extract_inferred_labels(self.test_data)

    @lru_cache(maxsize=1)
    def inferred_labels(self) -> Dict[str, NodeLabelling]:
        return self._extract_inferred_labels(self.data)
