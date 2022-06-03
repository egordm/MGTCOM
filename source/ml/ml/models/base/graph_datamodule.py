from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Union, Tuple

import pytorch_lightning as pl
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.typing import Metadata, NodeType

from datasets import GraphDataset
from datasets.transforms.homogenify import homogenify
from datasets.transforms.ensure_timestamps import EnsureTimestampsTransform
from datasets.transforms.eval_edge_split import EvalEdgeSplitTransform
from datasets.transforms.eval_node_split import EvalNodeSplitTransform
from datasets.utils.labels import LabelDict
from datasets.utils.types import Snapshots
from datasets.transforms.to_homogeneous import to_homogeneous
from ml.evaluation import extract_edge_prediction_pairs, EdgePredictionBatch
from ml.utils import HParams, DataLoaderParams, dict_catv
from shared import get_logger

logger = get_logger(Path(__file__).stem)

@dataclass
class GraphDataModuleParams(HParams):
    eval_inference: bool = False

    lp_max_pairs: int = 5000
    """Maximum number of pairs to use for link prediction."""
    train_on_full_data: bool = False
    """Whether to use the full dataset for training."""

    split_force: bool = False
    """Whether to force resplit the dataset. If false, predefined splits will be preferred."""
    split_num_val: float = 0.1
    """Fraction of the dataset to use for validation."""
    split_num_test: float = 0.1
    """Fraction of the dataset to use for testing."""

    homogenify: bool = False
    """Whether to convert the dataset to homogenous one before training."""


class GraphDataModule(pl.LightningDataModule):
    dataset: GraphDataset
    hparams: Union[GraphDataModuleParams, DataLoaderParams]
    loader_params: DataLoaderParams

    train_data: Union[HeteroData, Data]
    val_data: Union[HeteroData, Data]
    test_data: Union[HeteroData, Data]
    heterogenous: bool = True

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
        if self.hparams.train_on_full_data:
            logger.warning("Using full dataset for training. There is no validation or test set.")
            self.train_data, self.val_data, self.test_data = self.data, self.data, self.data
        else:
            if self.hparams.eval_inference:
                self.train_data, self.val_data, self.test_data = EvalNodeSplitTransform(
                    force_resplit=self.hparams.split_force,
                    num_val=self.hparams.split_num_val,
                    num_test=self.hparams.split_num_test,
                )(self.data)
            else:
                self.train_data, self.val_data, self.test_data = EvalEdgeSplitTransform(
                    force_resplit=self.hparams.split_force,
                    num_val=self.hparams.split_num_val,
                    num_test=self.hparams.split_num_test,
                    key_prefix='lp_'
                )(self.data)

        if self.hparams.homogenify:
            self.data, self.train_data, self.val_data, self.test_data = (
                homogenify(self.data),
                homogenify(self.train_data),
                homogenify(self.val_data),
                homogenify(self.test_data),
            )

        logger.info('=' * 80)
        logger.info(f'Using dataset {self.dataset.name}')
        logger.info(str(self.data))
        logger.info('=' * 80)

    @property
    def metadata(self) -> Metadata:
        return self.data.metadata()

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        return self.train_data.num_nodes_dict

    @property
    def snapshots(self) -> Optional[Dict[int, Snapshots]]:
        if isinstance(self.dataset, GraphDataset) and self.dataset.snapshots is not None:
            return self.dataset.snapshots
        return None

    def _edge_prediction_pairs(
        self, data: Union[HeteroData, Data], mask_name: str = 'train_mask'
    ) -> EdgePredictionBatch:
        """
        It takes a heterogeneous graph and returns a tuple of two tensors, the edges and the edge labels.

        :param data: HeteroData
        :type data: HeteroData
        :param mask_name: The name of the edge mask attribute, defaults to train_mask
        :type mask_name: str (optional)
        """
        prefix = 'lp_' if not self.hparams.eval_inference else ''

        if isinstance(data, HeteroData):
            hdata = to_homogeneous(
                data,
                node_attrs=[], edge_attrs=[f'{prefix}{mask_name}'],
                add_node_type=False, add_edge_type=False
            )
        else:
            hdata = data

        return extract_edge_prediction_pairs(
            hdata.edge_index, hdata.num_nodes, getattr(hdata, f'edge_{prefix}{mask_name}'),
            max_samples=self.hparams.lp_max_pairs
        )

    def link_prediction_pairs(self) -> Tuple[EdgePredictionBatch, EdgePredictionBatch, EdgePredictionBatch]:
        if self.hparams.eval_inference:
            return (
                self._edge_prediction_pairs(self.train_data, 'train_mask'),
                self._edge_prediction_pairs(self.val_data, 'val_mask'),
                self._edge_prediction_pairs(self.test_data, 'test_mask')
            )
        else:
            # Note: that the edges are taken from the full dataset
            # while subsets don't include these edges
            return (
                self._edge_prediction_pairs(self.data, 'train_mask'),
                self._edge_prediction_pairs(self.data, 'val_mask'),
                self._edge_prediction_pairs(self.data, 'test_mask')
            )

    def _extract_labels(
            self, data: HeteroData
    ) -> Dict[str, LabelDict]:
        node_labels = {}
        for label in self.dataset.labels():
            node_labels[label] = getattr(data, f'{label}_dict')

        return node_labels

    def train_labels_dict(self) -> Dict[str, LabelDict]:
        return self._extract_labels(self.train_data)

    @lru_cache(maxsize=1)
    def train_labels(self) -> Dict[str, Tensor]:
        return {
            label_name: dict_catv(labels)
            for label_name, labels in self.train_labels_dict().items()
        }

    def val_labels_dict(self) -> Dict[str, LabelDict]:
        return self._extract_labels(self.val_data)

    @lru_cache(maxsize=1)
    def val_labels(self) -> Dict[str, Tensor]:
        return {
            label_name: dict_catv(labels)
            for label_name, labels in self.val_labels_dict().items()
        }

    def test_labels_dict(self) -> Dict[str, LabelDict]:
        return self._extract_labels(self.test_data)

    @lru_cache(maxsize=1)
    def test_labels(self) -> Dict[str, Tensor]:
        return {
            label_name: dict_catv(labels)
            for label_name, labels in self.test_labels_dict().items()
        }

    def labels_dict(self) -> Dict[str, LabelDict]:
        return self._extract_labels(self.data)

    @lru_cache(maxsize=1)
    def labels(self) -> Dict[str, Tensor]:
        return {
            label_name: dict_catv(labels)
            for label_name, labels in self.labels_dict().items()
        }
