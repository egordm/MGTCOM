import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from cdlib import NodeClustering
from cdlib.classes.clustering import Clustering

from shared.graph import ComList, read_comlist, write_comlist, read_coms, coms_to_comlist, comlist_to_coms, write_coms, \
    Coms
from shared.logger import get_logger

NodeId = int

LOG = get_logger(os.path.basename(__file__))


@dataclass
class CommunityAssignment:
    data: ComList

    @classmethod
    def load_comlist(cls, filepath: str) -> 'CommunityAssignment':
        return CommunityAssignment(
            read_comlist(filepath)
        )

    def save_comlist(self, filepath: str) -> None:
        write_comlist(self.data, filepath)

    def to_comlist(self) -> ComList:
        return self.data

    @classmethod
    def load_comms(cls, filepath: str) -> 'CommunityAssignment':
        return CommunityAssignment(
            coms_to_comlist(read_coms(filepath))
        )

    def save_comms(self, filepath: str) -> None:
        write_coms(self.to_comms(), filepath)

    def to_comms(self) -> Coms:
        return comlist_to_coms(self.data)

    @classmethod
    def from_clustering(cls, clustering: Clustering) -> 'CommunityAssignment':
        return CommunityAssignment(
            coms_to_comlist(dict(enumerate(clustering.communities)))
        )

    def to_clustering(self) -> NodeClustering:
        return NodeClustering(
            list(self.to_comms().values()),
            None
        )

    def clone(self) -> 'CommunityAssignment':
        return CommunityAssignment(self.data.copy())

    def remap_nodes(self, nodemapping: pd.Series) -> 'CommunityAssignment':
        result = self.data.join(nodemapping, how='inner')
        if result.isnull().any().any():
            LOG.warning("Some nodes were not found in the nodemapping and thus have no community assignment.")
            result = result.dropna()

        result.set_index('gid', inplace=True)
        result.index.name = 'nid'

        return CommunityAssignment(result)

    def remap_communities(self) -> 'CommunityAssignment':
        com_names = self.data['cid'].unique()
        data = self.data.copy()
        data['cid'] = data['cid'].replace(com_names, range(len(com_names)))
        return CommunityAssignment(data)

    def overlapping(self):
        return len(self.data.index) > len(self.data.index.unique())

    def is_empty(self) -> bool:
        return len(self.data.index) == 0

    def community_count(self) -> int:
        return len(self.data['cid'].unique())
        # return int(self.data['cid'].max() + 1) if np.isfinite(self.data['cid'].max()) else 0

    def filter_nodes(self, gids: Iterable[NodeId]) -> 'CommunityAssignment':
        if not isinstance(gids, set):
            gids = set(gids)

        self.data = self.data[self.data.index.isin(gids)]
        return self

    def ensure_nodes(self, gids: Iterable[NodeId]) -> 'CommunityAssignment':
        if not isinstance(gids, set):
            gids = set(gids)

        if not gids.issubset(self.data.index):
            next_cid = self.community_count()
            missing_nodes = gids.difference(self.data.index)
            corresponding_cids = range(next_cid, next_cid + len(missing_nodes))

            df_missing = pd.DataFrame({
                'nid': list(missing_nodes),
                'cid': list(corresponding_cids)
            }).set_index('nid')

            self.data = self.data.append(df_missing)
            self.data.sort_index(inplace=True)

        return self

    def with_nodes(self, gids: Iterable[NodeId]) -> 'CommunityAssignment':
        self.filter_nodes(gids)
        self.ensure_nodes(gids)
        return self
