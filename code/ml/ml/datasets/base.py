from abc import abstractmethod
from typing import Optional, Callable, Dict

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset as THGInMemoryDataset, HeteroData
from torch_geometric.typing import NodeType, EdgeType

from shared.constants import DATASETS_DATA_TORCH


class InMemoryDataset(THGInMemoryDataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        if not root:
            root = str(DATASETS_DATA_TORCH.joinpath(self.name))

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def hetero_from_pandas(
        node_dfs: Dict[NodeType, pd.DataFrame],
        edge_dfs: Dict[EdgeType, pd.DataFrame],
) -> HeteroData:
    data = HeteroData()

    for node_type, df in node_dfs.items():
        store = data[node_type]
        features = [c for c in df.columns if c.startswith('feat_')]
        if len(features):
            store.x = torch.stack([torch.tensor(df[c].values) for c in features], dim=1).float()
        else:
            store.x = torch.zeros(df.shape[0])

        for c in df.columns:
            if c.startswith('feat_') or c.startswith('_'):
                continue

            setattr(store, c, torch.tensor(df[c].values))

    for edge_type, df in edge_dfs.items():
        store = data[edge_type]
        store.edge_index = torch.tensor(df[['src', 'dst']].values).t().contiguous().long()

        for c in df.columns:
            if c.startswith('src') or c.startswith('dst'):
                continue

            setattr(store, c, torch.tensor(df[c].values))

    return data
