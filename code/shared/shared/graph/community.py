import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from shared.graph import ComList, read_comlist, write_comlist, read_coms, coms_to_comlist, comlist_to_coms, write_coms, \
    Coms
from shared.logger import get_logger

NodeId = int

LOG = get_logger(os.path.basename(__file__))


@dataclass
class CommunityAssignment:
    data: ComList
    named: bool = False

    @classmethod
    def load_comlist(cls, filepath: str, named=False) -> 'CommunityAssignment':
        return CommunityAssignment(
            read_comlist(filepath, named=named),
            named
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
        write_comlist(self.data, filepath, named=self.named)

    def to_comms(self) -> Coms:
        return comlist_to_coms(self.data)

    def save_comms(self, filepath: str) -> None:
        write_coms(self.to_comms(), filepath)


