import copy
from pathlib import Path
from typing import Union, List

from torch_geometric.data import HeteroData, Data

from shared import get_logger

logger = get_logger(Path(__file__).stem)


def graph_clean_keys(data: Union[HeteroData, Data], key_whitelist: List[str]) -> Union[HeteroData, Data]:
    output = copy.copy(data)
    for store in output.stores:
        keys = [key for key in store.keys() if key not in key_whitelist]
        for key in keys:
            del store[key]

    return output
