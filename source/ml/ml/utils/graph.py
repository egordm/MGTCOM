from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from shared import get_logger

logger = get_logger(Path(__file__).stem)

NodeCountDict = Dict[NodeType, int]

NAN_TIMESTAMP = -1


def extract_attribute(
        data: HeteroData, key: str, edge_attr=False, warn=False, error=True
) -> Optional[Dict[NodeType, Union[Tensor, np.ndarray]]]:
    if key not in data.keys:
        if warn:
            logger.warning(f"{key} not in data.keys")
            return None
        elif error:
            raise KeyError(f"{key} not in data.keys")

    stores = data.node_stores if not edge_attr else data.edge_stores

    is_numpy = False
    dtype = None
    has_attr = False
    for store in stores:
        if key in store.keys() and ((not edge_attr and store.is_node_attr(key)) or (edge_attr and store.is_edge_attr(key))):
            has_attr = True
            is_numpy = isinstance(store[key], np.ndarray)
            dtype = store[key].dtype

    if not has_attr:
        if warn:
            logger.warning(f"{key} not in any {'edge' if edge_attr else 'node'} store")
            return None
        elif error:
            raise KeyError(f"{key} not in any {'edge' if edge_attr else 'node'} store")

    output = {}
    for store in stores:
        entity_type = store._key

        if key in store.keys():
            output[entity_type] = store[key]
        else:
            if warn:
                logger.warning(f"{key} not in {entity_type} store")
                if is_numpy:
                    output[entity_type] = np.full([store.num_nodes], np.nan, dtype=dtype)
                else:
                    output[entity_type] = torch.tensor([store.num_nodes], dtype=dtype).fill_(np.nan)

    return output



