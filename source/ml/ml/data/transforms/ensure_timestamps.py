from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from ml.utils.graph import NAN_TIMESTAMP
from shared import get_logger

logger = get_logger(Path(__file__).stem)


class EnsureTimestampsTransform(BaseTransform):
    """
    It ensures that all node and edge stores have a `timestamp_from` field
    """
    def __init__(self, warn: bool = False):
        self.warn = warn

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        > If the timestamp_from attribute is not present in the node or edge store, then create a new attribute called
        timestamp_from and fill it with NaN values

        :param data: HeteroData
        :type data: HeteroData
        :return: A hetero data object with the timestamp_from attribute added to the node and edge stores.
        """
        for store in data.node_stores:
            if 'timestamp_from' not in store.keys():
                if self.warn:
                    logger.warning(f'No timestamp_from defined for node_type: {store._key}')
                store.timestamp_from = torch.full([store.num_nodes], fill_value=NAN_TIMESTAMP, dtype=torch.long)

        for store in data.edge_stores:
            if 'timestamp_from' not in store.keys():
                if self.warn:
                    logger.warning(f'No timestamp_from defined for edge_type: {store._key}')
                store.timestamp_from = torch.full([store.num_edges], fill_value=NAN_TIMESTAMP, dtype=torch.long)

        return data
