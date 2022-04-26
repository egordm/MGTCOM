from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.cli import _Registry
from torch import Tensor
from torch_geometric.data import InMemoryDataset as THGInMemoryDataset, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage, BaseStorage
from torch_geometric.typing import Metadata

from datasets.transforms.random_node_split import RandomNodeSplit
from datasets.utils.labels import extract_louvain_labels, extract_timestamp_labels, extract_snapshot_labels
from datasets.utils.types import Snapshots
from shared import get_logger
from shared.paths import CACHE_PATH

logger = get_logger(Path(__file__).stem)

DATASET_REGISTRY = _Registry()


class BaseGraphDataset(THGInMemoryDataset):
    name: str = ''
    description: str = ''
    tags: List[str] = []

    data: HeteroData
    snapshots: Dict[int, Snapshots] = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @property
    def metadata(self) -> Metadata:
        return self.data.metadata()

    @staticmethod
    def labels() -> List[str]:
        return []


class GraphDataset(BaseGraphDataset):
    def __init__(
            self,
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            num_val: float = 0.1,
            num_test: float = 0.1,
    ):
        self.num_val = num_val
        self.num_test = num_test

        if not root:
            root = str(CACHE_PATH.joinpath('dataset', self.name))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.after_load()

    def after_load(self):
        pass

    def _process_node(self, data: HeteroData, store: NodeStorage, df: pd.DataFrame):
        df = df.set_index('id').sort_index()
        assert len(df) == df.index.max() - df.index.min() + 1, 'Duplicate IDs'

        features = [c for c in df.columns if c.startswith('feat_')]
        if len(features):
            store.x = torch.stack([torch.tensor(df[c].astype(float).fillna(0).values) for c in features], dim=1).float()
        else:
            store.x = torch.zeros(df.shape[0])

        if 'name' in df.columns:
            store.name = np.array(df['name'].values)

        self._process_timestamps(store, df)

    def _process_edge(self, data: HeteroData, store: EdgeStorage, df: pd.DataFrame):
        store.edge_index = torch.vstack([
            torch.tensor(df['src'].values).long(),
            torch.tensor(df['dst'].values).long(),
        ])
        src, _, dst = store._key

        assert data[src].num_nodes > store.edge_index[0, :].max(), 'Invalid src'
        assert data[dst].num_nodes > store.edge_index[1, :].max(), 'Invalid dst'

        self._process_timestamps(store, df)

    def _process_timestamps(self, store: BaseStorage, df: pd.DataFrame):
        cols = ['timestamp_from', 'timestamp_to']
        for col in cols:
            if col in df:
                if str(df[col].dtype).startswith('datetime'):
                    df[col] = df[col].apply(lambda x: x.timestamp() if not pd.isnull(x) else np.NAN)

                setattr(store, col, torch.tensor(df[col].fillna(-1).values).long())

        if hasattr(store, 'timestamp_from') and not hasattr(store, 'timestamp_to'):
            setattr(store, 'timestamp_to', torch.full([len(df)], -1).long())

    def _process_graph(self, files: List[str]) -> HeteroData:
        data = HeteroData()

        for raw_path in files:
            entity_type, name = Path(raw_path).name.split('__')
            type = tuple(name.split('_')) if '_' in name else name
            store = data[type]

            df = pd.read_parquet(raw_path, engine='pyarrow', use_nullable_dtypes=True)
            if entity_type == 'node':
                self._process_node(data, store, df)
            elif entity_type == 'edge':
                self._process_edge(data, store, df)
            else:
                raise ValueError(f'Unknown entity type: {entity_type}')

        return data

    @abstractmethod
    def _preprocess(self):
        pass

    def _extract_labels(self, data: HeteroData):
        logger.info('Extracting Louvain labels')
        louvain = extract_louvain_labels(data)
        for node_type, labels in louvain.items():
            data[node_type].louvain = labels

        if self.snapshots is not None:
            logger.info('Extracting node timestamps')
            node_timestamps = extract_timestamp_labels(data)
            for i, snapshot in self.snapshots.items():
                logger.info(f'Extracting timestamps for snapshot {i}')
                snapshot_labels = extract_snapshot_labels(node_timestamps, snapshot)
                for node_type, labels in snapshot_labels.items():
                    setattr(data[node_type], f'label_snapshot_{i}', labels)

        return data

    def process(self):
        data = self._preprocess()
        data = self._extract_labels(data)

        data = RandomNodeSplit(
            split="train_rest",
            num_splits=1,
            num_val=self.num_val,
            num_test=self.num_test,
            key=None,
        )(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
