import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from cdlib import NodeClustering

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

    @classmethod
    def load_comms(cls, filepath: str) -> 'CommunityAssignment':
        return CommunityAssignment(
            coms_to_comlist(read_coms(filepath))
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

    def renumber_communities(self) -> 'CommunityAssignment':
        com_names = self.data['cid'].unique()
        data = self.data.copy()
        data['cid'] = data['cid'].replace(com_names, range(len(com_names)))
        return CommunityAssignment(data)

    def filter_nodes(self, gid: List[NodeId]) -> 'CommunityAssignment':
        data = self.data.copy()
        data = data[data.index.isin(gid)]
        return CommunityAssignment(data)

    def to_comlist(self) -> ComList:
        return self.data

    def save_comlist(self, filepath: str) -> None:
        write_comlist(self.data, filepath)

    def to_comms(self) -> Coms:
        return comlist_to_coms(self.data)

    def save_comms(self, filepath: str) -> None:
        write_coms(self.to_comms(), filepath)

    def overlapping(self):
        return len(self.data.index) > len(self.data.index.unique())

    def is_empty(self) -> bool:
        return len(self.data.index) == 0

    def to_nodeclustering(self) -> NodeClustering:
        return NodeClustering(
            list(self.to_comms().values()),
            None
        )

    def add_missing_nodes(self, node_count: int):
        max_community = int(self.data['cid'].max() + 1) if np.isfinite(self.data['cid'].max()) else 0
        missing_nodes = set(range(node_count)).difference(set(self.data.index))
        corresponding_cids = range(max_community, max_community + len(missing_nodes))

        df_missing = pd.DataFrame({
            'nid': list(missing_nodes),
            'cid': list(corresponding_cids)
        }).set_index('nid')

        self.data = self.data.append(df_missing)
        self.data.sort_index(inplace=True)

    def community_count(self) -> int:
        return len(self.data['cid'].unique())
